
import json
import logging
import os
import tempfile
import urllib.request
from typing import Generator, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.config import (
    COARSE_FINE_BOUNDARY, COARSE_RES, DEFAULT_MODEL_URL,
    DEFAULT_VOCAB_URL, DEVICE, FALLBACK_PROMPTS, FINE_RES, RING_WIDTH,
)

logger = logging.getLogger("train_tspo")

_WANDB = False


def wandb_init(cfg, project, name):
    global _WANDB
    try:
        import wandb
        wandb.init(project=project, name=name, config=cfg)
        _WANDB = True
        logger.info(f"[W&B] {wandb.run.url}")
    except Exception as e:
        logger.warning(f"[W&B] disabled: {e}")


def wandb_log(d, step):
    if not _WANDB:
        return
    try:
        import wandb
        wandb.log(d, step=step)
    except Exception:
        pass


def wandb_finish():
    if not _WANDB:
        return
    try:
        import wandb
        wandb.finish()
    except Exception:
        pass


def ensure_artifact(local_path, download_url, name):
    if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    url = download_url.replace("/blob/", "/resolve/")
    logger.info(f"{name} not found. Downloading from {url}")
    fd, tmp = tempfile.mkstemp()
    os.close(fd)
    try:
        urllib.request.urlretrieve(url, tmp)
        if os.path.getsize(tmp) == 0:
            raise RuntimeError(f"Empty download: {name}")
        os.replace(tmp, local_path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return local_path

def build_ring_mask(mask, ring_width=RING_WIDTH):
    m    = (mask > 0).astype(np.uint8)
    k    = ring_width * 2 + 1
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return (cv2.dilate(m, kern) - cv2.erode(m, kern)).astype(bool)


def mask_bounding_box(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        h, w = mask.shape
        return 0, h, 0, w
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return int(y0), int(y1) + 1, int(x0), int(x1) + 1


def crop_to_mask(img_np, mask):
    y0, y1, x0, x1 = mask_bounding_box(mask)
    return img_np[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def pil_to_np(img, size=512):
    return np.array(img.convert("RGB").resize((size, size)))


def np_to_pil(arr):
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def denorm(x, lo, hi):
    return lo + x * (hi - lo)


def alpha_schedule(t_norm):
    return 0.3 + 0.6 * t_norm


def tau_policy(t_norm):
    return 0.40 + 0.25 * t_norm


def tau_faithfulness(t_norm):
    if t_norm < 0.85:
        return 0.30 + 0.30 * t_norm
    return 0.55 - 0.10 * (t_norm - 0.85) / 0.15


def _img_transform(res):
    return transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


IMG_T512 = _img_transform(512)


def build_prompt_iterator(
    dataset_name: str,
    split: str,
    seed: int = 42,
    buf: int = 5000,
) -> Generator[Tuple[str, int], None, None]:
    try:
        from datasets import load_dataset
        hf  = load_dataset(dataset_name, split=split, streaming=True)
        hf  = hf.shuffle(seed=seed, buffer_size=buf)
        col = next(
            (c for c in hf.column_names
             if any(k in c.lower() for k in ("prompt", "caption", "text"))),
            hf.column_names[0],
        )

        def _gen():
            idx = 0
            while True:
                for row in hf:
                    txt = str(row.get(col, "") or "").strip()
                    if txt:
                        yield txt, (seed + idx) % 100_000
                    idx += 1

        return _gen()

    except Exception as e:
        logger.warning(f"HF load failed ({e}). Using fallback prompts.")

        def _fb():
            idx = 0
            while True:
                yield FALLBACK_PROMPTS[idx % len(FALLBACK_PROMPTS)], (seed + idx) % 100_000
                idx += 1

        return _fb()

def encode_prompt_safe(pipe, prompt, device):
    try:
        pe, npe = pipe.encode_prompt(prompt, device, 1, True)
        return torch.cat([npe, pe])
    except TypeError:
        return pipe._encode_prompt(prompt, device, 1, True)
