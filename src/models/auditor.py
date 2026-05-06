
import logging

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.config import (
    COARSE_FINE_BOUNDARY, COARSE_RES, DEFAULT_MODEL_URL, DEFAULT_VOCAB_URL,
    DEVICE, FINE_RES, HARM_CLASSES, NUM_CLASSES, RING_WIDTH, TEXT_DIM,
)
from src.models.text_encoder import SimpleTextEncoder, SimpleTokenizer
from src.utils.helpers import (
    build_ring_mask, crop_to_mask, ensure_artifact, np_to_pil, pil_to_np,
    _img_transform,
)

logger = logging.getLogger("train_tspo")

class CompleteMultiTaskAuditor(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, vocab_size=50000):
        super().__init__()
        resnet        = models.resnet101(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        self.text_encoder = SimpleTextEncoder(vocab_size=vocab_size)
        self.adv_head     = nn.Conv2d(2048, 1, 1)
        self.class_head   = nn.Conv2d(2048, num_classes, 1)
        self.quality_head = nn.Conv2d(2048, 1, 1)
        self.image_proj   = nn.Conv2d(2048, TEXT_DIM, 1)
        self.query_norm   = nn.LayerNorm(TEXT_DIM)
        self.key_norm     = nn.LayerNorm(TEXT_DIM)

        self.cross_attention = nn.MultiheadAttention(TEXT_DIM, num_heads=8, batch_first=True)
        self.img_proj_head   = nn.Sequential(
            nn.Linear(TEXT_DIM, 256), nn.ReLU(), nn.Linear(256, 256),
        )
        self.txt_proj_head   = nn.Sequential(
            nn.Linear(TEXT_DIM, 256), nn.ReLU(), nn.Linear(256, 256),
        )
        self.policy_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, num_classes),
        )
        self.seam_feat = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(512),
        )
        self.seam_cls = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(256), nn.Conv2d(256, 1, 1),
        )

        self.timestep_embed = nn.Sequential(
            nn.Linear(1, 128), nn.SiLU(),
            nn.Linear(128, 256), nn.SiLU(),
            nn.Linear(256, 512),
        )
        self.film_gamma = nn.Linear(512, 2048)
        self.film_beta  = nn.Linear(512, 2048)

    def _film(self, feats, ts_emb):
        B = feats.size(0)
        return (
            (1.0 + self.film_gamma(ts_emb).view(B, 2048, 1, 1)) * feats
            + self.film_beta(ts_emb).view(B, 2048, 1, 1)
        )

    def forward(self, x, text_tokens=None, timestep=None):
        B     = x.size(0)
        feats = self.features(x)
        if timestep is not None:
            feats = self._film(feats, self.timestep_embed(timestep))

        adv_map       = torch.sigmoid(self.adv_head(feats))
        risk_maps     = torch.sigmoid(self.class_head(feats))
        seam_map      = torch.sigmoid(self.seam_cls(self.seam_feat(feats)))
        policy_logits = self.policy_head(feats)

        img_embed = txt_embed = None
        faithfulness = torch.zeros(B, 1, device=x.device)

        if text_tokens is not None:
            text_feat, seq_feat, pad_mask = self.text_encoder(text_tokens)
            img_seq = self.image_proj(feats).view(B, TEXT_DIM, -1).permute(0, 2, 1)
            att_seq, _ = self.cross_attention(
                self.query_norm(img_seq),
                self.key_norm(seq_feat),
                self.key_norm(seq_feat),
                key_padding_mask=pad_mask,
            )
            img_embed    = F.normalize(self.img_proj_head(att_seq.mean(1)), dim=-1)
            txt_embed    = F.normalize(self.txt_proj_head(text_feat),       dim=-1)
            faithfulness = (
                (F.cosine_similarity(img_embed, txt_embed, dim=-1) + 1.0) / 2.0
            ).unsqueeze(1)

        adv_prob    = torch.sigmoid(F.adaptive_avg_pool2d(adv_map, (1, 1)).flatten(1))
        policy_safe = 1.0 - adv_prob
        seam_score  = F.adaptive_avg_pool2d(seam_map, (1, 1)).flatten(1)

        return dict(
            faithfulness=faithfulness,
            policy_safe=policy_safe,
            seam_score=seam_score,
            adv_map=adv_map,
            risk_maps=risk_maps,
            policy_logits=policy_logits,
            img_embed=img_embed,
            txt_embed=txt_embed,
        )

class AuditorWrapper:
    def __init__(self, model_path, vocab_path, heatmap_percentile=65):
        model_path = ensure_artifact(model_path, DEFAULT_MODEL_URL, "auditor")
        vocab_path = ensure_artifact(vocab_path, DEFAULT_VOCAB_URL,  "vocab")

        self.tokenizer          = SimpleTokenizer(vocab_path)
        vocab_size              = len(self.tokenizer.word_to_idx)
        self.heatmap_percentile = heatmap_percentile

        self.model = CompleteMultiTaskAuditor(NUM_CLASSES, vocab_size)
        # BUG 4 FIX: strict=False so checkpoint architecture differences don't crash
        self.model.load_state_dict(
            torch.load(model_path, map_location=DEVICE), strict=False,
        )
        self.model.to(DEVICE).eval()

    def _run(self, pil, tokens, t_norm, res=224):
        tf = _img_transform(res)
        t  = tf(pil.convert("RGB")).unsqueeze(0).to(DEVICE)
        ts = torch.tensor([[t_norm]], device=DEVICE, dtype=torch.float32)
        with torch.no_grad():
            return self.model(t, text_tokens=tokens, timestep=ts)

    def _make_mask(self, out):
        pred_idx = int(
            torch.argmax(F.softmax(out["policy_logits"][0], dim=0)).item()
        )
        raw_map  = out["risk_maps"][0, pred_idx]
        heatmap  = F.interpolate(
            raw_map.unsqueeze(0).unsqueeze(0).float(),
            size=(512, 512), mode="bilinear", align_corners=False,
        )[0, 0].detach().cpu().numpy()

        thresh    = np.percentile(heatmap, self.heatmap_percentile)
        feathered = cv2.GaussianBlur(
            (heatmap >= thresh).astype(np.float32), (15, 15), 5,
        )
        mask_512  = (feathered > 0.5).astype(np.uint8)
        return mask_512, build_ring_mask(mask_512), heatmap

    def audit(self, image, prompt="", t_norm=0.0, mask=None):
        if isinstance(image, np.ndarray):
            arr = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image
            pil = np_to_pil(arr)
        else:
            pil = image

        tokens = self.tokenizer.encode(prompt).unsqueeze(0).to(DEVICE)
        res    = FINE_RES if t_norm >= COARSE_FINE_BOUNDARY else COARSE_RES
        out    = self._run(pil, tokens, t_norm, res)

        F_       = out["faithfulness"][0, 0].item()
        P        = out["policy_safe"][0, 0].item()
        B        = out["seam_score"][0, 0].item()
        pred_idx = int(torch.argmax(F.softmax(out["policy_logits"][0], dim=0)).item())
        mask_512, ring, heatmap = self._make_mask(out)

        crop_mask = mask if mask is not None else mask_512
        img_np    = pil_to_np(pil, 512)
        crop_img, _ = crop_to_mask(img_np, crop_mask)

        F_R = P_R = B_R = 0.0
        if crop_img.size > 0:
            out_R = self._run(np_to_pil(crop_img), tokens, t_norm, COARSE_RES)
            F_R   = out_R["faithfulness"][0, 0].item()
            P_R   = out_R["policy_safe"][0, 0].item()
            B_R   = out_R["seam_score"][0, 0].item()

        return dict(
            F=F_, P=P, B=B,
            F_R=F_R, P_R=P_R, B_R=B_R,
            harm_category=HARM_CLASSES[pred_idx],
            adversarial_score=1.0 - P,
            mask_512=mask_512,
            ring_mask=ring,
            adv_heatmap_512=heatmap,
            img_embed=(
                out["img_embed"][0].detach().cpu()
                if out["img_embed"] is not None else torch.zeros(256)
            ),
            class_idx=pred_idx,
            tokens=tokens,
        )


class ScoreCalibrator:
    def __init__(self):
        self._raw:  list = []
        self._wins: list = []
        self.delta = 0.05 

    def record(self, margin, win):
        self._raw.append(float(margin))
        self._wins.append(1.0 if win else 0.0)

    def recalibrate(self):
        if len(self._raw) < 50:
            return self.delta
        try:
            from sklearn.isotonic import IsotonicRegression
            import numpy as np

            idx  = np.argsort(self._raw)
            r    = np.array(self._raw)[idx]
            w    = np.array(self._wins)[idx]
            ir   = IsotonicRegression(out_of_bounds="clip", increasing=True)
            ir.fit(r, w)
            test  = np.linspace(0, max(r), 500)
            probs = ir.predict(test)
            valid = np.where(probs >= 0.9)[0]
            if len(valid) > 0:
                self.delta = float(test[valid[0]])
                logger.info(f"[Calib] δ = {self.delta:.4f}")
        except ImportError:
            pass
        return self.delta
