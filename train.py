import json
import logging
import os
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from diffusers import DDIMScheduler, StableDiffusionPipeline
from src.config import (
    AUDIT_EVERY_N_STEPS, AUDIT_START_FRAC, DELTA_MARGIN, DEVICE, DTYPE,
    STATE_DIM, parse_args,
)
from src.models import (
    AuditorWrapper, Inpainter, ListwiseJudgeRefiner, ScoreCalibrator,
    StateEncoder, StateEncoderWrapper, TSPOPolicy, TSPOTrainer,
)
from src.utils import (
    build_prompt_iterator, compute_guard_utility, encode_prompt_safe,
    np_to_pil, seam_quality_vgg, visualise_tournament,
    wandb_finish, wandb_init, wandb_log,
)
from src.utils.helpers import _img_transform

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("train_tspo")

def run_diffusion_with_tournaments(
    base_pipe, prompt, num_steps, seed,
    auditor, inpainter, policy, state_enc,
    N, adv_threshold, calibrator, vis_dir, vis_every, step_global,
):
    sched = base_pipe.scheduler
    sched.set_timesteps(num_steps)
    gen   = torch.Generator(str(DEVICE)).manual_seed(seed)

    with torch.no_grad():
        text_emb = encode_prompt_safe(base_pipe, prompt, str(DEVICE)).to(dtype=DTYPE)

    latents = (
        torch.randn(
            (1, base_pipe.unet.config.in_channels, 64, 64),
            generator=gen, device=str(DEVICE), dtype=DTYPE,
        ) * sched.init_noise_sigma
    )

    total    = len(sched.timesteps)
    start_at = int(AUDIT_START_FRAC * total)
    entries: List[dict] = []
    n_audited = [0]

    def decode(z):
        zf  = z.to(DTYPE) / base_pipe.vae.config.scaling_factor
        with torch.no_grad():
            img = base_pipe.vae.decode(zf).sample
        img = (img.float().clamp(-1, 1) + 1) / 2
        return Image.fromarray(
            (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )

    for step_idx, t in enumerate(sched.timesteps):
        t_norm = t.item() / 1000.0
        lat_in = sched.scale_model_input(torch.cat([latents] * 2), t)

        with torch.no_grad():
            noise_pred = base_pipe.unet(lat_in, t, encoder_hidden_states=text_emb).sample

        u, c       = noise_pred.chunk(2)
        noise_pred = u + 7.5 * (c - u)
        latents    = sched.step(noise_pred, t, latents, generator=gen).prev_sample

        if step_idx < start_at:
            continue
        if (step_idx - start_at) % AUDIT_EVERY_N_STEPS != 0:
            continue

        image = decode(latents)
        ctrl  = auditor.audit(image, prompt, t_norm=t_norm)

        if ctrl["adversarial_score"] < adv_threshold:
            entries.append({"skipped": True})
            continue

        n_audited[0] += 1
        mask  = ctrl["mask_512"]
        ring  = ctrl["ring_mask"]

        state     = state_enc.encode(
            prompt=prompt, latent=latents.float(),
            img_embed=ctrl["img_embed"], mask_512=mask, t_norm=t_norm,
        )
        knob_sets  = policy.sample_knobs(state, N=N)
        candidates = inpainter.generate_candidates(image, mask, prompt, knob_sets)

        utilities, cand_scores, cand_embeds = [], [], []
        for cand in candidates:
            # BUG 5 FIX: real VGG-LPIPS seam quality, not L1 pixel distance
            B_i = seam_quality_vgg(image, cand, ring)
            cs  = auditor.audit(cand, prompt, t_norm=t_norm, mask=mask)
            u_i = compute_guard_utility(cs["P_R"], cs["F_R"], B_i, t_norm)
            utilities.append(u_i)
            cand_scores.append(cs)
            cand_embeds.append(cs["img_embed"])

        best_idx = int(np.argmax(utilities))
        accepted = utilities[best_idx] > 0.0

        calibrator.record(utilities[best_idx], accepted)

        if accepted:
            wk      = knob_sets[best_idx]
            z_edit  = inpainter.short_ddim_inversion(
                candidates[best_idx], wk.inversion_depth, t_norm,
            )
            latents = inpainter.blend_latents(
                latents.float(), z_edit, mask, t_norm,
            ).to(DTYPE)

        if vis_dir and vis_every > 0 and n_audited[0] % vis_every == 0:
            try:
                path = visualise_tournament(
                    prompt, image, ctrl, candidates, cand_scores,
                    utilities, step_global, t_norm, vis_dir, accepted,
                )
                wandb_log(
                    {f"vis/{'success' if accepted else 'fail'}": path}, step_global,
                )
            except Exception as ve:
                logger.warning(f"Vis error: {ve}")

        entries.append(dict(
            state=state.cpu(),
            knob_sets=knob_sets,
            utilities=utilities,
            cand_embeds=cand_embeds,
            accepted=accepted,
            images=[image] + candidates,
            tokens=ctrl["tokens"],
            t_norm=t_norm,
        ))

    return latents, entries

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.no_wandb:
        wandb_init(
            vars(args),
            args.wandb_project,
            args.wandb_run_name or f"tspo_N{args.N}_lr{args.lr}",
        )
    auditor   = AuditorWrapper(args.auditor_model, args.auditor_vocab)
    inpainter = Inpainter(inpainter_pth=args.inpainter_pth)

    state_enc = StateEncoderWrapper(
        encoder=StateEncoder(),
        tokenizer=auditor.tokenizer,
        text_encoder=auditor.model.text_encoder,
    )

    policy = TSPOPolicy(state_dim=STATE_DIM).to(DEVICE)
    if args.resume:
        policy.load_state_dict(torch.load(args.resume, map_location=DEVICE))
        logger.info(f"Resumed from '{args.resume}'")

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    trainer   = TSPOTrainer(
        policy=policy,
        optimizer=optimizer,
        batch_size=args.batch_size,
        N=args.N,
        lambda_entropy_cont=args.lambda_ent_cont,
        lambda_entropy_disc=args.lambda_ent_disc,
        lambda_compute=args.lambda_compute,
        lambda_diversity=args.lambda_div,
    )

    calibrator = ScoreCalibrator()

    judge_opt = torch.optim.Adam(
        [p for p in auditor.model.parameters() if p.requires_grad],
        lr=args.lr_judge,
    )
    judge = ListwiseJudgeRefiner(
        auditor_model=auditor.model,
        optimizer=judge_opt,
        batch_size=args.judge_batch,
        lambda_ln=args.lambda_listnet,
        lambda_pl=args.lambda_pl,
    )
    base_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=DTYPE,
        safety_checker=None,
        requires_safety_checker=False,
    )
    base_pipe.safety_checker    = None
    base_pipe.feature_extractor = None
    base_pipe.scheduler = DDIMScheduler.from_config(base_pipe.scheduler.config)
    base_pipe = base_pipe.to(str(DEVICE))
    base_pipe.enable_attention_slicing()

    prompt_iter = build_prompt_iterator(
        args.dataset, args.split, args.seed, args.shuffle_buffer,
    )

    log_path = os.path.join(args.output_dir, "tournament_log.jsonl")
    log_fh   = open(log_path, "a")
    stats    = dict(
        total=0, skipped=0, accepted=0,
        policy_updates=0, judge_updates=0,
    )
    
    pending_prompt: Optional[Tuple[str, int]] = None
    pbar = tqdm(range(args.steps), desc="TSPO", dynamic_ncols=True)

    for step in pbar:
        if pending_prompt is not None:
            prompt, seed = pending_prompt
            pending_prompt = None
        else:
            prompt, seed = next(prompt_iter)

        stats["total"] += 1   
        log_entries: List[dict] = []
        for attempt in range(args.max_retries):
            try:
                _, entries = run_diffusion_with_tournaments(
                    base_pipe=base_pipe,
                    prompt=prompt,
                    num_steps=args.diffusion_steps,
                    seed=seed + attempt,
                    auditor=auditor,
                    inpainter=inpainter,
                    policy=policy,
                    state_enc=state_enc,
                    N=args.N,
                    adv_threshold=args.adv_threshold,
                    calibrator=calibrator,
                    vis_dir=args.vis_dir,
                    vis_every=args.vis_every,
                    step_global=step,
                )
            except Exception as e:
                logger.warning(f"Step {step} attempt {attempt}: {e}", exc_info=True)
                entries = []

            log_entries = entries
            any_won = any(
                e.get("accepted") for e in entries if not e.get("skipped")
            )
            if any_won:
                break
            if attempt < args.max_retries - 1:
                logger.info(
                    f"[-] No winner (step {step}, attempt {attempt + 1}). Retrying."
                )
                any_accepted = False
        for entry in log_entries:
            if entry.get("skipped"):
                stats["skipped"] += 1
                continue
            accepted = entry.get("accepted", False)
            if accepted:
                any_accepted = True
                stats["accepted"] += 1
            log_fh.write(json.dumps({
                "step": step,
                "prompt": prompt[:120],
                "utilities": entry.get("utilities", []),
                "accepted": accepted,
            }) + "\n")
            log_fh.flush()
            trainer.accumulate(entry)
            judge.accumulate(
                images   =entry.get("images", []),
                tokens   =entry.get("tokens", torch.zeros(1, 77, dtype=torch.long)),
                utilities=entry.get("utilities", []),
                t_norm   =entry.get("t_norm", 0.0),
            )

        step_metrics: Dict[str, float] = {}
        if trainer.ready():
            info = trainer.step()
            stats["policy_updates"] += 1
            step_metrics.update({f"tspo/{k}": v for k, v in info.items()})
            logger.info(
                f"Step {step:4d} | policy_upd={stats['policy_updates']} "
                f"loss={info.get('total', 0):.4f}  pg={info.get('pg_loss', 0):.4f}  "
                f"div={info.get('diversity', 0):.4f}  "
                f"acc={stats['accepted']}/{stats['total'] - stats['skipped']}"
            )

        if judge.ready():
            jinfo = judge.step()
            stats["judge_updates"] += 1
            step_metrics.update({f"judge/{k}": v for k, v in jinfo.items()})
            logger.info(
                f"  judge_upd={stats['judge_updates']} "
                f"judge_loss={jinfo.get('judge_loss', 0):.4f}"
            )

        if args.recalib_every > 0 and step > 0 and step % args.recalib_every == 0:
            new_delta = calibrator.recalibrate()
            step_metrics["calib/delta"] = new_delta
            logger.info(f"[Calib] δ = {new_delta:.4f}")

        step_metrics.update({
            "stats/total":          stats["total"],
            "stats/skipped":        stats["skipped"],
            "stats/accepted":       stats["accepted"],
            "stats/acc_rate":       stats["accepted"] / max(1, stats["total"] - stats["skipped"]),
            "stats/policy_updates": stats["policy_updates"],
            "stats/judge_updates":  stats["judge_updates"],
        })
        wandb_log(step_metrics, step=step)

        pbar.set_postfix(
            acc=stats["accepted"],
            buf=len(trainer._buffer),
            p_upd=stats["policy_updates"],
            j_upd=stats["judge_updates"],
        )

        if step > 0 and step % args.save_every == 0:
            torch.save(
                policy.state_dict(),
                os.path.join(args.output_dir, f"tspo_step{step:05d}.pth"),
            )
            torch.save(
                state_enc.encoder.state_dict(),
                os.path.join(args.output_dir, f"state_enc_step{step:05d}.pth"),
            )
            logger.info(f"Checkpoint @ step {step}")

    pbar.close()
    log_fh.close()

    torch.save(policy.state_dict(),
               os.path.join(args.output_dir, "tspo_final.pth"))
    torch.save(state_enc.encoder.state_dict(),
               os.path.join(args.output_dir, "state_enc_final.pth"))
    logger.info(f"Done. Stats: {stats}")
    wandb_log({"final/" + k: v for k, v in stats.items()}, step=args.steps)
    wandb_finish()


if __name__ == "__main__":
    main()
