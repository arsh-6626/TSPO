
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.helpers import np_to_pil, pil_to_np


def visualise_tournament(
    prompt, control, ctrl_s, candidates, cand_scores,
    utilities, step, t_norm, out_dir, accepted,
):

    nc   = len(candidates)
    ncol = nc + 2
    fig, axes = plt.subplots(3, ncol, figsize=(4 * ncol, 11))
    best = int(np.argmax(utilities))

    def _overlay(img, mask):
        a  = pil_to_np(img, 512).copy().astype(float)
        ov = a.copy()
        ov[mask > 0] = [255, 0, 0]
        return np_to_pil((0.4 * ov + 0.6 * a).astype(np.uint8))

    def _plot_col(col, img, sc, u, tag):
        axes[0, col].imshow(pil_to_np(img, 512))
        axes[0, col].set_title(f"{tag}\nu={u:.3f}", fontsize=8)
        axes[1, col].imshow(_overlay(img, sc["mask_512"]))
        axes[1, col].set_title(sc.get("harm_category", ""), fontsize=8)
        axes[2, col].imshow(pil_to_np(img, 512))
        axes[2, col].imshow(
            sc["adv_heatmap_512"], cmap="jet", alpha=0.5, vmin=0, vmax=1,
        )
        axes[2, col].set_title("heatmap", fontsize=8)

    _plot_col(0, control, ctrl_s, 0.0, "Control")
    for i in range(nc):
        label = ("★ " if (i == best and accepted) else "") + f"C{i + 1}"
        _plot_col(i + 1, candidates[i], cand_scores[i], utilities[i], label)

    if accepted:
        _plot_col(ncol - 1, candidates[best], cand_scores[best], utilities[best], "✓ Accepted")
    else:
        _plot_col(ncol - 1, control, ctrl_s, 0.0, "✗ No winner")

    for ax in axes.flatten():
        ax.axis("off")

    status = "SUCCESS" if accepted else "FAIL"
    plt.suptitle(
        f"[{status}] step={step} t={t_norm:.2f} | {prompt[:70]}", fontsize=8,
    )
    plt.tight_layout()

    sub  = "success" if accepted else "fail"
    path = os.path.join(out_dir, sub, f"step{step:05d}_t{t_norm:.2f}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return path
