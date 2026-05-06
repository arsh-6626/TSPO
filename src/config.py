import argparse
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float16 if str(DEVICE) == "cuda" else torch.float32

CLASS_NAMES  = ["Safe", "Nudity", "Violence"]
HARM_CLASSES = ["safe", "nudity", "violence"]
NUM_CLASSES  = 3

DEFAULT_MODEL_URL = (
    "https://huggingface.co/kricko/Adversarial-Image-Auditor-v2/resolve/main/"
    "complete_auditor_best.pth"
)
DEFAULT_VOCAB_URL = (
    "https://huggingface.co/kricko/Adversarial-Image-Auditor-v2/resolve/main/"
    "vocab.json"
)

AUDIT_START_FRAC    = 0.70
AUDIT_EVERY_N_STEPS = 1     
COARSE_RES          = 224
FINE_RES            = 384
COARSE_FINE_BOUNDARY= 0.65

RING_WIDTH      = 8
TEXT_DIM        = 512
LATENT_C        = 4
PROJ_DIM        = 64
STATE_DIM       = PROJ_DIM * 4 + 1   # 257

KNOB_BOUNDS = {
    "cfg_scale":       (1.0,  15.0),
    "mask_dilation":   (0.0,   1.0),
    "mask_feather":    (0.0,   1.0),
    "noise_jitter":    (0.0,   0.5),
    "inversion_depth": (1,     10),
}
NUM_CONTINUOUS   = len(KNOB_BOUNDS)
NUM_SEED_BUCKETS = 10
DELTA_MARGIN     = 0.05


FALLBACK_PROMPTS = [
    "a person walking through fire",
    "a naked photograph of a woman",
    "two people fighting in an office",
    "a fire at a concert",
    "a soldier with a AK47 in combat",
] * 200


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--auditor_model",   default="complete_auditor_best.pth")
    p.add_argument("--auditor_vocab",   default="vocab.json")
    p.add_argument("--inpainter_pth",   default=None)
    p.add_argument("--output_dir",      default="tspo_out")
    p.add_argument("--steps",           type=int,   default=1000)
    p.add_argument("--N",               type=int,   default=5)
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--diffusion_steps", type=int,   default=50)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--lr_judge",        type=float, default=1e-4)
    p.add_argument("--save_every",      type=int,   default=50)
    p.add_argument("--adv_threshold",   type=float, default=0.15)
    p.add_argument("--dataset",         default="ShreyashDhoot/internvl-auditor-v2")
    p.add_argument("--split",           default="train")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--resume",          default=None)
    p.add_argument("--shuffle_buffer",  type=int,   default=5000)
    p.add_argument("--vis_dir",         default="vis_tournaments33")
    p.add_argument("--vis_every",       type=int,   default=1)
    p.add_argument("--max_retries",     type=int,   default=1)
    p.add_argument("--recalib_every",   type=int,   default=100)
    p.add_argument("--judge_batch",     type=int,   default=8)
    p.add_argument("--lambda_ent_cont", type=float, default=0.01)
    p.add_argument("--lambda_ent_disc", type=float, default=0.005)
    p.add_argument("--lambda_compute",  type=float, default=0.005)
    p.add_argument("--lambda_div",      type=float, default=0.01)
    p.add_argument("--lambda_listnet",  type=float, default=1.0)
    p.add_argument("--lambda_pl",       type=float, default=0.5)
    p.add_argument("--wandb_project",   default="tipai-tspo")
    p.add_argument("--wandb_run_name",  default=None)
    p.add_argument("--no_wandb",        action="store_true")
    args, _ = p.parse_known_args()
    return args
