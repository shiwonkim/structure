"""Figure 2 — Per-anchor CAP attention heatmap.

Visualises how individual learned anchors attend to different spatial regions,
demonstrating the interpretability of BA's Cross-Attention Pooling.

Grid: 3-4 rows (images) x (1 + N_anchors) columns.
  Column 0 = original image.
  Columns 1..N = heatmap overlay for the selected anchors, captioned with the
  anchor index and the class most associated with that anchor.

Usage:
    PYTHONPATH=. python scripts/viz/anchor_attention_heatmap.py \
        --config  configs/ba/vits_minilm/token_k128.yaml \
        --ckpt    results/alignment-.../checkpoint-epoch588.pth \
        --gpu 0 \
        [--indices 15,100,400,850] \
        [--n-anchors 5] \
        [--output results/figures/anchor_attention_heatmap.png] \
        [--pickle results/figures/_anchor_heatmap_cache.pkl]
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from torchvision.datasets import VOCSegmentation

from scripts.viz.viz_utils import (
    DatasetSpec,
    anchor_class_correspondence,
    build_text_features,
    compute_anchor_attention,
    extract_patch_features,
    load_full_pipeline,
    overlay_heatmap,
)
from src.evaluation.zero_shot_segmentation import VOC2012_CLASS_PROMPTS, VOC2012_CLASSES


LAYER_IMG = 11
LAYER_TXT = 6
DEFAULT_INDICES = [15, 100, 400, 850]


def select_interesting_anchors(
    attn: torch.Tensor,
    correspondence: list,
    n: int = 5,
) -> List[int]:
    """Pick anchors with highest spatial concentration (low entropy → sharp attention).

    Parameters
    ----------
    attn : (P, K) — per-anchor attention distribution over patches.
    correspondence : list of (anchor_idx, class_name, activation).
    n : number of anchors to select.
    """
    K = attn.shape[1]
    # Spatial concentration = negative entropy of the attention column.
    # Lower entropy → more focused → more interesting.
    eps = 1e-10
    entropy = -(attn * (attn + eps).log()).sum(dim=0)  # (K,)

    # Also filter out anchors that map to "background" — less interesting.
    bg_anchors = {c[0] for c in correspondence if c[1] == "background"}

    # Sort by ascending entropy (most concentrated first), skip bg.
    order = entropy.argsort().tolist()
    selected = []
    for k in order:
        if k in bg_anchors:
            continue
        selected.append(k)
        if len(selected) >= n:
            break
    # Fallback: if too few non-bg anchors, fill with bg ones.
    if len(selected) < n:
        for k in order:
            if k not in selected:
                selected.append(k)
            if len(selected) >= n:
                break
    return selected


def render_figure(
    dataset,
    indices: List[int],
    pipeline: dict,
    text_feats: torch.Tensor,
    correspondence: list,
    n_anchors: int,
    output_path: str,
    pickle_path: Optional[str] = None,
):
    device = pipeline["device"]
    rows_data = []

    for idx in indices:
        pil_img, pil_mask = dataset[idx]
        orig = np.array(pil_img.convert("RGB"))
        H_orig, W_orig = orig.shape[:2]

        feats = extract_patch_features(
            pil_img,
            pipeline["vision_model"],
            pipeline["image_transform"],
            LAYER_IMG,
            device,
        )
        attn, h = compute_anchor_attention(feats, pipeline["alignment_image"], device)
        # attn: (P, K), h: patch grid side

        selected = select_interesting_anchors(attn, correspondence, n=n_anchors)

        anchor_panels = []
        for k in selected:
            attn_k = attn[:, k].cpu().numpy()  # (P,)
            attn_map = attn_k.reshape(h, h)
            # Normalise to [0, 1] for colourmap
            vmin, vmax = attn_map.min(), attn_map.max()
            if vmax > vmin:
                attn_map = (attn_map - vmin) / (vmax - vmin)
            else:
                attn_map = np.zeros_like(attn_map)
            # Upsample to original image size
            attn_up = F.interpolate(
                torch.from_numpy(attn_map).float().unsqueeze(0).unsqueeze(0),
                size=(H_orig, W_orig),
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy()
            panel = overlay_heatmap(orig, attn_up, alpha=0.5, cmap="inferno")
            cls_name = correspondence[k][1]
            anchor_panels.append({
                "image": panel,
                "anchor_idx": k,
                "class_name": cls_name,
                "activation": correspondence[k][2],
            })

        rows_data.append({
            "idx": idx,
            "original": orig,
            "anchors": anchor_panels,
        })

    if pickle_path:
        with open(pickle_path, "wb") as f:
            pickle.dump(rows_data, f)
        logger.info(f"Cached intermediate arrays to {pickle_path}")

    _plot(rows_data, n_anchors, output_path)


def _plot(rows_data: list, n_anchors: int, output_path: str):
    n_rows = len(rows_data)
    n_cols = 1 + n_anchors

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.0 * n_cols, 3.2 * n_rows),
        gridspec_kw={"wspace": 0.03, "hspace": 0.12},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for r, row in enumerate(rows_data):
        ax = axes[r, 0]
        ax.imshow(row["original"])
        ax.set_xticks([])
        ax.set_yticks([])
        if r == 0:
            ax.set_title("Original", fontsize=11, fontweight="bold")

        for c, ap in enumerate(row["anchors"]):
            ax = axes[r, c + 1]
            ax.imshow(ap["image"])
            ax.set_xticks([])
            ax.set_yticks([])
            caption = f"Anchor {ap['anchor_idx']}\n\"{ap['class_name']}\""
            if r == 0:
                ax.set_title(caption, fontsize=9, fontweight="bold")
            else:
                ax.set_title(caption, fontsize=9)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    logger.info(f"Saved anchor attention figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Figure 2: per-anchor CAP attention heatmap")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--indices", default=None, help="Comma-separated VOC val indices")
    parser.add_argument("--n-anchors", type=int, default=5)
    parser.add_argument("--output", default="results/figures/anchor_attention_heatmap.png")
    parser.add_argument("--pickle", default="results/figures/_anchor_heatmap_cache.pkl")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pipeline = load_full_pipeline(args.config, args.ckpt, device)

    dataset = VOCSegmentation(
        root="data/pascal_voc", year="2012", image_set="val", download=False,
    )
    prompt_names = [VOC2012_CLASS_PROMPTS[c] for c in VOC2012_CLASSES]

    logger.info("Building text features (ensemble) for anchor-class correspondence...")
    text_feats = build_text_features(
        classnames=prompt_names, strategy="ensemble",
        tokenizer=pipeline["tokenizer"],
        language_model=pipeline["language_model"],
        layer_txt=LAYER_TXT,
        alignment_text=pipeline["alignment_text"],
        device=device, token_level=True,
    )
    text_feats_n = F.normalize(text_feats.to(device), dim=-1)

    correspondence = anchor_class_correspondence(text_feats_n, VOC2012_CLASSES)
    logger.info("Anchor-class correspondence (top 10 by activation):")
    top10 = sorted(correspondence, key=lambda x: -x[2])[:10]
    for k, cls, act in top10:
        logger.info(f"  Anchor {k:3d} -> {cls:15s}  (act={act:.3f})")

    indices = (
        [int(x) for x in args.indices.split(",")]
        if args.indices else DEFAULT_INDICES
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    render_figure(
        dataset, indices, pipeline,
        text_feats_n, correspondence,
        n_anchors=args.n_anchors,
        output_path=args.output,
        pickle_path=args.pickle,
    )


if __name__ == "__main__":
    main()
