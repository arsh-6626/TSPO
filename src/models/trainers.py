import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import DEVICE
from src.utils.helpers import _img_transform

logger = logging.getLogger("train_tspo")


class TSPOTrainer:
    def __init__(self, policy, optimizer, batch_size=8, N=5, **lambdas):
        self.policy    = policy
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.N          = N
        self.lambdas    = lambdas
        self._buffer: List[dict] = []
        self.total_updates = 0

    def accumulate(self, entry):
        if not entry.get("skipped"):
            self._buffer.append(entry)

    def ready(self):
        return len(self._buffer) >= self.batch_size

    def step(self):
        batch        = self._buffer[: self.batch_size]
        self._buffer = self._buffer[self.batch_size :]

        all_s, all_c, all_sd, all_u, all_e = [], [], [], [], []
        for entry in batch:
            s   = entry["state"].to(DEVICE)
            ks  = entry["knob_sets"]
            us  = entry["utilities"]
            emb = entry.get("cand_embeds", [None] * self.N)
            for k, e in zip(ks, emb):
                all_s.append(s)
                all_c.append(k.raw_cont)
                all_sd.append(k.seed_offset // 100)
                all_e.append(e if e is not None else torch.zeros(256))
            all_u.extend(us)

        if not all_s:
            return {}

        self.optimizer.zero_grad()
        loss, info = self.policy.tspo_loss(
            torch.stack(all_s),
            torch.tensor(all_c,  dtype=torch.float32, device=DEVICE),
            torch.tensor(all_sd, dtype=torch.long,    device=DEVICE),
            torch.tensor(all_u,  dtype=torch.float32, device=DEVICE),
            N=self.N,
            cand_embeds=torch.stack([e.to(DEVICE) for e in all_e]),
            **self.lambdas,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        self.total_updates += 1
        return info

class ListwiseJudgeRefiner:

    def __init__(
        self,
        auditor_model,
        optimizer,
        batch_size=8,
        lambda_ln=1.0,
        lambda_pl=0.5,
        tau_model=1.0,
        tau_target=0.5,
        device=None,
    ):
        self.model      = auditor_model
        self.optimizer  = optimizer
        self.batch_size = batch_size
        self.lambda_ln  = lambda_ln
        self.lambda_pl  = lambda_pl
        self.tau_model  = tau_model
        self.tau_target = tau_target
        self.device     = device or DEVICE
        self._buffer: List[dict] = []
        self.total_updates = 0

        trainable = (
            "img_proj_head", "txt_proj_head",
            "cross_attention", "query_norm", "key_norm",
        )
        for name, p in self.model.named_parameters():
            p.requires_grad_(any(k in name for k in trainable))

    def accumulate(self, images, tokens, utilities, t_norm):
        if len(images) == len(utilities) > 1:
            self._buffer.append(dict(
                images=images,
                tokens=tokens.cpu(),
                utilities=utilities,
                t_norm=t_norm,
            ))

    def ready(self):
        return len(self._buffer) >= self.batch_size

    def step(self):
        batch        = self._buffer[: self.batch_size]
        self._buffer = self._buffer[self.batch_size :]

        self.model.train()
        self.optimizer.zero_grad()
        total, n = torch.tensor(0.0, device=self.device), 0
        _tf = _img_transform(224)

        for entry in batch:
            imgs    = entry["images"]
            tokens  = entry["tokens"].to(self.device)
            utils_t = torch.tensor(entry["utilities"], dtype=torch.float32, device=self.device)
            t_norm  = entry["t_norm"]
            N1      = len(imgs)

            imgs_t = torch.stack([_tf(img.convert("RGB")) for img in imgs]).to(self.device)
            ts_t   = torch.tensor([[t_norm]] * N1, device=self.device)
            tok_b  = tokens.expand(N1, -1)

            out    = self.model(imgs_t, text_tokens=tok_b, timestep=ts_t)
            # Use policy_safe as the ranking signal (replaces removed score_head)
            scores = out["policy_safe"].squeeze(-1)
            winner = int(torch.argmax(utils_t).item())

            # ListNet
            q    = F.softmax(utils_t / self.tau_target, dim=0)
            p    = F.log_softmax(scores / self.tau_model,  dim=0)
            l_ln = -(q * p).sum()

            # Plackett-Luce top-1
            l_pl = F.cross_entropy(
                scores.unsqueeze(0),
                torch.tensor([winner], device=self.device),
            )

            total = total + self.lambda_ln * l_ln + self.lambda_pl * l_pl
            n    += 1

        if n == 0:
            self.model.eval()
            return {}

        (total / n).backward()
        nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0,
        )
        self.optimizer.step()
        self.model.eval()
        self.total_updates += 1
        return dict(judge_loss=(total / n).item(), judge_updates=self.total_updates)
