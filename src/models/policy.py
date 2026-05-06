
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    DEVICE, KNOB_BOUNDS, LATENT_C, NUM_CONTINUOUS, NUM_SEED_BUCKETS,
    PROJ_DIM, STATE_DIM, TEXT_DIM,
)
from src.utils.helpers import denorm

@dataclass
class KnobSet:
    cfg_scale:       float
    mask_dilation:   float
    mask_feather:    float
    noise_jitter:    float
    inversion_depth: int
    seed_offset:     int
    raw_cont: list = field(default_factory=list)
    log_prob: float = 0.0


class TSPOPolicy(nn.Module):
    def __init__(
        self,
        state_dim=STATE_DIM,
        hidden_dims=(256, 128, 64),
        log_std_min=-4.0,
        log_std_max=0.5,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers, d = [], state_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU()]
            d = h

        self.trunk        = nn.Sequential(*layers)
        self.mean_head    = nn.Linear(d, NUM_CONTINUOUS)
        self.log_std_head = nn.Linear(d, NUM_CONTINUOUS)
        self.seed_head    = nn.Linear(d, NUM_SEED_BUCKETS)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, s):
        h = self.trunk(s)
        return (
            torch.sigmoid(self.mean_head(h)),
            self.log_std_head(h).clamp(self.log_std_min, self.log_std_max),
            self.seed_head(h),
        )

    def sample_knobs(self, state, N=5) -> List[KnobSet]:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.expand(N, -1)
        mean, log_std, seed_logits = self.forward(state)
        std = log_std.exp()
        out = []
        for i in range(N):
            raw = (mean[i] + std[i] * torch.randn_like(mean[i])).clamp(0, 1)
            c   = raw.tolist()
            sd  = torch.distributions.Categorical(logits=seed_logits[i])
            st  = sd.sample()
            nd  = torch.distributions.Normal(mean[i], std[i])
            lp  = (nd.log_prob(raw).sum() + sd.log_prob(st)).item()
            out.append(KnobSet(
                cfg_scale      =denorm(c[0], *KNOB_BOUNDS["cfg_scale"]),
                mask_dilation  =denorm(c[1], *KNOB_BOUNDS["mask_dilation"]),
                mask_feather   =denorm(c[2], *KNOB_BOUNDS["mask_feather"]),
                noise_jitter   =denorm(c[3], *KNOB_BOUNDS["noise_jitter"]),
                inversion_depth=max(1, int(round(denorm(c[4], *KNOB_BOUNDS["inversion_depth"])))),
                seed_offset    =int(st.item() * 100),
                raw_cont=c,
                log_prob=lp,
            ))
        return out

    def tspo_loss(
        self,
        states, raw_conts, seed_buckets, utilities, N,
        lambda_entropy_cont=0.01,
        lambda_entropy_disc=0.005,
        lambda_compute=0.005,
        lambda_diversity=0.01,
        cand_embeds=None,
        **_,
    ):
        mean, log_std, seed_logits = self.forward(states)
        std = log_std.exp()
        B   = states.shape[0] // N

        utils_2d = utilities.view(B, N)
        tau      = utils_2d.std(dim=1, keepdim=True).clamp(min=1e-6)
        w_2d     = F.softmax(utils_2d / tau, dim=1) - 1.0 / N
        w_cands  = w_2d.reshape(-1)

        nd        = torch.distributions.Normal(mean, std)
        lp_cont   = nd.log_prob(raw_conts.clamp(0, 1)).sum(-1)
        sd        = torch.distributions.Categorical(logits=seed_logits)
        log_probs = lp_cont + sd.log_prob(seed_buckets)

        pg_loss  = -(w_cands.detach() * log_probs).sum()
        ent_cont = nd.entropy().sum(-1).mean()
        ent_disc = sd.entropy().mean()
        cost     = raw_conts[:, 4].mean()

        if cand_embeds is not None and cand_embeds.shape[0] == B * N:
            f2d = cand_embeds.view(B, N, -1)
            div, pairs = torch.tensor(0.0, device=states.device), 0
            for i in range(N):
                for j in range(i + 1, N):
                    div += F.pairwise_distance(f2d[:, i], f2d[:, j]).mean()
                    pairs += 1
            diversity = div / max(pairs, 1)
        else:
            f2d = raw_conts.view(B, N, -1)
            pw  = torch.cdist(f2d, f2d, p=2)
            msk = ~torch.eye(N, dtype=torch.bool, device=pw.device).unsqueeze(0)
            diversity = pw[msk.expand_as(pw)].mean()

        loss = (
            pg_loss
            - lambda_entropy_cont * ent_cont
            - lambda_entropy_disc * ent_disc
            + lambda_compute      * cost
            - lambda_diversity    * diversity
        )
        return loss, dict(
            pg_loss=pg_loss.item(),
            ent_cont=ent_cont.item(),
            ent_disc=ent_disc.item(),
            cost=cost.item(),
            diversity=diversity.item(),
            total=loss.item(),
        )

class StateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_proj   = nn.Linear(TEXT_DIM, PROJ_DIM)
        self.latent_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(4 * 4 * LATENT_C, PROJ_DIM), nn.ReLU(),
        )
        self.image_proj = nn.Linear(256, PROJ_DIM)
        self.mask_proj  = nn.Linear(1, PROJ_DIM)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, text_embed, latent, image_embed, mask_mean, t_norm):
        p  = F.relu(self.text_proj(text_embed))
        z  = F.relu(self.latent_proj(latent))
        im = F.relu(self.image_proj(image_embed))
        m  = F.relu(self.mask_proj(mask_mean))
        return torch.cat([p, z, im, m, t_norm], dim=-1)   # (B, STATE_DIM)

class StateEncoderWrapper:
    def __init__(self, encoder: StateEncoder, tokenizer, text_encoder: nn.Module):
        assert text_encoder is not None, "text_encoder must be provided"
        self.encoder      = encoder.to(DEVICE)
        self.tokenizer    = tokenizer
        self.text_encoder = text_encoder.to(DEVICE)
        self.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

    def encode(self, prompt, latent, img_embed, mask_512, t_norm):
        tokens = self.tokenizer.encode(prompt).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            text_embed, _, _ = self.text_encoder(tokens)   # (1, TEXT_DIM) — never zeros

        latent_b  = latent.to(DEVICE).float()
        img_b     = img_embed.unsqueeze(0).to(DEVICE).float()
        mask_mean = torch.tensor([[float(mask_512.mean())]], device=DEVICE)
        t_t       = torch.tensor([[t_norm]], device=DEVICE, dtype=torch.float32)

        with torch.no_grad():
            state = self.encoder(text_embed, latent_b, img_b, mask_mean, t_t)
        return state.squeeze(0)   # (STATE_DIM,)
