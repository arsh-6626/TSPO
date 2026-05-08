# TSPO — Tournament-Based Safe Policy Optimization

Part of the **TIPAI-TSPO** project under PRAGYA AI LAB, BITS GOA. Trains a knob-selection policy over a frozen Stable Diffusion inpainting pipeline. At flagged diffusion timesteps, the policy proposes candidate hyperparameter sets, runs them through a safety auditor, and updates via policy gradient on the results.

## How it works

At each audited timestep:
1. **Auditor** scores the current frame — if `adversarial_score ≥ threshold`, a tournament runs
2. **Policy** samples `N` candidate `KnobSet`s (cfg scale, mask dilation/feather, noise jitter, inversion depth)
3. **Inpainter** generates `N` candidates using those knobs
4. Each candidate is scored: `U = policy_safe × faithfulness × seam_quality`
5. The winner is blended back into the latent; all candidates update the policy

## Setup

```bash
pip install torch torchvision diffusers transformers accelerate
pip install opencv-python-headless pillow numpy tqdm matplotlib datasets
pip install wandb lpips scikit-learn  # optional
```

Auditor weights and vocab download automatically from HuggingFace on first run.

## Usage

```bash
# basic
python train.py

# custom
python train.py --steps 500 --N 5 --batch_size 4 --lr 3e-4 --output_dir runs/exp1

# resume
python train.py --resume runs/exp1/tspo_step00200.pth --no_wandb
```

## Key arguments

| Argument | Default | Description |
|---|---|---|
| `--steps` | `1000` | Training steps |
| `--N` | `5` | Candidates per tournament |
| `--batch_size` | `4` | Steps buffered before a policy update |
| `--diffusion_steps` | `50` | DDIM steps per run |
| `--lr` / `--lr_judge` | `3e-4` / `1e-4` | Policy / judge learning rates |
| `--adv_threshold` | `0.15` | Minimum adversarial score to trigger a tournament |
| `--max_retries` | `1` | Retry diffusion if no winner found |
| `--recalib_every` | `100` | Steps between isotonic δ recalibration |
| `--save_every` | `50` | Checkpoint frequency |
| `--no_wandb` | `False` | Disable W&B logging |

## Structure

```
train.py                   # entry point + diffusion/tournament loop
src/
  config.py                # constants, knob bounds, parse_args()
  models/
    auditor.py             # ResNet-101 multi-task auditor + calibrator
    policy.py              # TSPOPolicy, StateEncoder, KnobSet
    inpainter.py           # SD inpainting wrapper + DDIM inversion
    text_encoder.py        # LSTM tokenizer + text encoder
    trainers.py            # TSPOTrainer, ListwiseJudgeRefiner
  utils/
    helpers.py             # image/mask ops, transforms, prompt iterator
    metrics.py             # VGG-LPIPS seam quality, guard utility
    visualise.py           # tournament grid visualisation
```

## Outputs

```
<output_dir>/
  tspo_step00050.pth       # policy checkpoints
  state_enc_step00050.pth  # state encoder checkpoints
  tournament_log.jsonl     # per-step results
<vis_dir>/
  success/                 # visualisation grids — accepted candidates
  fail/                    # visualisation grids — no winner
```
