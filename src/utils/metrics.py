
import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from src.config import DELTA_MARGIN, DEVICE
from src.utils.helpers import tau_faithfulness, tau_policy

logger = logging.getLogger("train_tspo")

VGG_CACHE = None


def get_vgg():
    global VGG_CACHE
    if VGG_CACHE is None:
        vgg    = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        feats  = list(vgg.features.children())
        slices = nn.ModuleList([nn.Sequential(*feats[:e]) for e in [4, 9, 16, 23, 30]])
        chans  = [64, 128, 256, 512, 512]
        lins   = nn.ModuleList([nn.Conv2d(c, 1, 1, bias=False) for c in chans])
        for lin in lins:
            nn.init.constant_(lin.weight, 1.0 / lin.in_channels)
        for p in slices.parameters():
            p.requires_grad_(False)
        VGG_CACHE = (slices.to(DEVICE).eval(), lins.to(DEVICE).eval())
    return VGG_CACHE


def lpips_tensor(img_pil, size=256):
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return tf(img_pil.convert("RGB")).unsqueeze(0).to(DEVICE)


def ring_t(ring, h, w):
    r = cv2.resize(ring.astype(np.float32), (w, h), cv2.INTER_LINEAR)
    return (
        torch.from_numpy((r > 0.3).astype(np.float32))
        .unsqueeze(0).unsqueeze(0)
        .to(DEVICE)
    )


def seam_quality_vgg(ctrl_pil, cand_pil, ring, kappa=5.0):
    """B_i = exp(-κ · LPIPS_{∂m}(cand, ctrl)) via VGG multi-scale features."""
    if not ring.any():
        return 1.0
    try:
        import lpips as _lpips_lib

        fn = getattr(seam_quality_vgg, "_lpips_fn", None)
        if fn is None:
            fn = _lpips_lib.LPIPS(net="vgg", spatial=True).to(DEVICE).eval()
            for p in fn.parameters():
                p.requires_grad_(False)
            seam_quality_vgg._lpips_fn = fn

        t0 = lpips_tensor(ctrl_pil)
        t1 = lpips_tensor(cand_pil)
        with torch.no_grad():
            d = fn(t0, t1).squeeze()
        h, w = d.shape
        rt   = ring_t(ring, h, w).squeeze()
        n    = rt.sum()
        if n < 1:
            return 1.0
        return float(torch.exp(-kappa * (d * rt).sum() / n).item())

    except ImportError:
        pass
    slices, lins = get_vgg()
    t0 = lpips_tensor(ctrl_pil)
    t1 = lpips_tensor(cand_pil)
    with torch.no_grad():
        h0, h1 = t0, t1
        total, count = 0.0, 0
        for sl, lin in zip(slices, lins):
            h0 = sl(h0)
            h1 = sl(h1)
            f0 = F.normalize(h0, dim=1)
            f1 = F.normalize(h1, dim=1)
            diff  = (f0 - f1) ** 2
            d_map = lin(diff).squeeze()
            fh, fw = d_map.shape
            rt    = ring_t(ring, fh, fw).squeeze()
            n     = rt.sum()
            if n < 1:
                continue
            total += ((d_map * rt).sum() / n).item()
            count += 1

    return float(np.exp(-kappa * total / max(count, 1)))


def compute_guard_utility(P_i, F_i, B_i, t_norm, delta=DELTA_MARGIN):
    policy_ok = float(P_i >= tau_policy(t_norm))
    faith_ok  = float(F_i >= tau_faithfulness(t_norm))
    return policy_ok * faith_ok * B_i
