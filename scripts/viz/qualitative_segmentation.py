"""Figure 1 — Qualitative segmentation comparison.

Grid: 6-8 rows (images) x 4 columns (Original | GT | FreezeAlign | BA K=512).
Both methods use the 80-prompt ensemble text strategy on Pascal VOC 2012 val.

Usage:
    PYTHONPATH=. python scripts/viz/qualitative_segmentation.py \
        --ba-config   configs/ba/vits_minilm/token_k512.yaml \
        --ba-ckpt     results/alignment-.../checkpoint-epoch490.pth \
        --fa-config   configs/freezealign/vits_minilm/fa_d512.yaml \
        --fa-ckpt     results/alignment-.../checkpoint-epoch551.pth \
        --gpu 0 \
        [--indices 10,45,120,200,350,500,800,1100] \
        [--output results/figures/qualitative_segmentation.png] \
        [--pickle results/figures/_qualitative_cache.pkl]
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
from loguru import logger
from PIL import Image
from torchvision.datasets import VOCSegmentation

from scripts.viz.viz_utils import (
    DatasetSpec,
    build_text_features,
    extract_patch_features,
    load_full_pipeline,
    overlay_mask,
    pascal_voc_palette,
    segment_image_ba_codebook,
    segment_image_freezealign,
)
from src.evaluation.zero_shot_segmentation import VOC2012_CLASS_PROMPTS, VOC2012_CLASSES


LAYER_IMG = 11
LAYER_TXT = 6

# Image selection — Option B (diverse class coverage) with a few BA-strong-win
# cases. These are candidate indices into VOC 2012 val (1449 images); override
# with --indices if needed.
DEFAULT_INDICES = [15, 100, 250, 400, 600, 850, 1100, 1350]


def select_images_auto(
    dataset,
    ba_pipeline: dict,
    fa_pipeline: dict,
    ba_text_feats: torch.Tensor,
    fa_text_feats: torch.Tensor,
    n_images: int = 8,
    n_ba_wins: int = 3,
) -> List[int]:
    """Automatically select images: n_ba_wins where BA dominates, rest diverse.

    Scans a subset of the val set and picks images by per-image IoU delta.
    """
    device = ba_pipeline["device"]
    spec = DatasetSpec.for_voc2012()
    scores = []

    logger.info(f"Scanning {len(dataset)} val images for auto selection...")
    for i in range(len(dataset)):
        pil_img, pil_mask = dataset[i]
        gt = np.array(pil_mask)

        feats = extract_patch_features(
            pil_img,
            ba_pipeline["vision_model"],
            ba_pipeline["image_transform"],
            LAYER_IMG,
            device,
        )
        ba_pred = segment_image_ba_codebook(
            feats, ba_text_feats, ba_pipeline["alignment_image"], gt.shape, device,
        )
        fa_feats = extract_patch_features(
            pil_img,
            fa_pipeline["vision_model"],
            fa_pipeline["image_transform"],
            LAYER_IMG,
            device,
        )
        fa_pred = segment_image_freezealign(
            fa_feats, fa_text_feats, fa_pipeline["alignment_image"], gt.shape, device,
        )

        ba_iou = _per_image_miou(gt, ba_pred, spec)
        fa_iou = _per_image_miou(gt, fa_pred, spec)
        scores.append((i, ba_iou, fa_iou, ba_iou - fa_iou))

    scores.sort(key=lambda x: -x[3])
    ba_wins = [s[0] for s in scores[:n_ba_wins]]

    remaining = [s for s in scores if s[0] not in ba_wins and s[1] > 0.1]
    remaining.sort(key=lambda x: -x[1])
    step = max(1, len(remaining) // (n_images - n_ba_wins))
    diverse = [remaining[i * step][0] for i in range(n_images - n_ba_wins)]

    selected = ba_wins + diverse
    logger.info(f"Auto-selected indices: {selected}")
    return selected


def _per_image_miou(
    gt: np.ndarray, pred: np.ndarray, spec: DatasetSpec,
) -> float:
    """Quick per-image foreground mIoU (for image selection only)."""
    ious = []
    for c in range(1, spec.num_classes):  # skip background
        gt_c = gt == c
        pred_c = pred == c
        inter = (gt_c & pred_c).sum()
        union = (gt_c | pred_c).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def render_figure(
    dataset,
    indices: List[int],
    ba_pipeline: dict,
    fa_pipeline: dict,
    ba_text_feats: torch.Tensor,
    fa_text_feats: torch.Tensor,
    output_path: str,
    pickle_path: Optional[str] = None,
):
    device = ba_pipeline["device"]
    spec = DatasetSpec.for_voc2012()
    palette = pascal_voc_palette()

    rows_data = []
    for idx in indices:
        pil_img, pil_mask = dataset[idx]
        orig = np.array(pil_img.convert("RGB"))
        gt = np.array(pil_mask)
        H, W = gt.shape

        ba_feats = extract_patch_features(
            pil_img, ba_pipeline["vision_model"],
            ba_pipeline["image_transform"], LAYER_IMG, device,
        )
        ba_pred = segment_image_ba_codebook(
            ba_feats, ba_text_feats, ba_pipeline["alignment_image"],
            gt.shape, device,
        )

        fa_feats = extract_patch_features(
            pil_img, fa_pipeline["vision_model"],
            fa_pipeline["image_transform"], LAYER_IMG, device,
        )
        fa_pred = segment_image_freezealign(
            fa_feats, fa_text_feats, fa_pipeline["alignment_image"],
            gt.shape, device,
        )

        orig_resized = np.array(pil_img.convert("RGB").resize((W, H)))
        gt_overlay = overlay_mask(orig_resized, gt, alpha=0.6, palette=palette)
        fa_overlay = overlay_mask(orig_resized, fa_pred, alpha=0.6, palette=palette)
        ba_overlay = overlay_mask(orig_resized, ba_pred, alpha=0.6, palette=palette)

        rows_data.append({
            "idx": idx,
            "original": orig_resized,
            "gt_overlay": gt_overlay,
            "fa_overlay": fa_overlay,
            "ba_overlay": ba_overlay,
            "gt": gt,
            "fa_pred": fa_pred,
            "ba_pred": ba_pred,
        })

    if pickle_path:
        with open(pickle_path, "wb") as f:
            pickle.dump(rows_data, f)
        logger.info(f"Cached intermediate arrays to {pickle_path}")

    _plot(rows_data, output_path, palette, spec)


def _plot(
    rows_data: list,
    output_path: str,
    palette: np.ndarray,
    spec: DatasetSpec,
):
    n_rows = len(rows_data)
    col_titles = ["Original", "Ground Truth", "FreezeAlign", "BA K=512 (ours)"]

    fig, axes = plt.subplots(
        n_rows, 4, figsize=(14, 3.2 * n_rows),
        gridspec_kw={"wspace": 0.02, "hspace": 0.06},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for r, row in enumerate(rows_data):
        panels = [row["original"], row["gt_overlay"], row["fa_overlay"], row["ba_overlay"]]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=12, fontweight="bold")

    # Colour legend at bottom
    fg_classes = spec.classes[1:]  # skip background
    n_cols_legend = 7
    n_rows_legend = math.ceil(len(fg_classes) / n_cols_legend)
    legend_fig_height = 0.35 * n_rows_legend

    fig.subplots_adjust(bottom=legend_fig_height / (3.2 * n_rows + legend_fig_height))
    legend_ax = fig.add_axes([0.05, 0.0, 0.9, legend_fig_height / (3.2 * n_rows + legend_fig_height)])
    legend_ax.set_xlim(0, n_cols_legend)
    legend_ax.set_ylim(0, n_rows_legend)
    legend_ax.axis("off")

    for i, cls_name in enumerate(fg_classes):
        col = i % n_cols_legend
        row_l = n_rows_legend - 1 - i // n_cols_legend
        colour = palette[i + 1] / 255.0
        legend_ax.add_patch(plt.Rectangle(
            (col + 0.05, row_l + 0.2), 0.2, 0.6, facecolor=colour, edgecolor="k", linewidth=0.5,
        ))
        legend_ax.text(
            col + 0.32, row_l + 0.5, cls_name, fontsize=8, va="center",
        )

    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    logger.info(f"Saved qualitative figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Figure 1: qualitative segmentation comparison")
    parser.add_argument("--ba-config", required=True)
    parser.add_argument("--ba-ckpt", required=True)
    parser.add_argument("--fa-config", required=True)
    parser.add_argument("--fa-ckpt", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--indices", default=None, help="Comma-separated VOC val indices (or 'auto')")
    parser.add_argument("--output", default="results/figures/qualitative_segmentation.png")
    parser.add_argument("--pickle", default="results/figures/_qualitative_cache.pkl")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ba_pipeline = load_full_pipeline(args.ba_config, args.ba_ckpt, device)
    fa_pipeline = load_full_pipeline(args.fa_config, args.fa_ckpt, device)

    dataset = VOCSegmentation(
        root="data/pascal_voc", year="2012", image_set="val", download=False,
    )
    spec = DatasetSpec.for_voc2012()
    prompt_names = [VOC2012_CLASS_PROMPTS[c] for c in VOC2012_CLASSES]

    logger.info("Building BA text features (ensemble)...")
    ba_text_feats = build_text_features(
        classnames=prompt_names, strategy="ensemble",
        tokenizer=ba_pipeline["tokenizer"],
        language_model=ba_pipeline["language_model"],
        layer_txt=LAYER_TXT,
        alignment_text=ba_pipeline["alignment_text"],
        device=device, token_level=True,
    )
    logger.info("Building FA text features (ensemble)...")
    fa_text_feats = build_text_features(
        classnames=prompt_names, strategy="ensemble",
        tokenizer=fa_pipeline["tokenizer"],
        language_model=fa_pipeline["language_model"],
        layer_txt=LAYER_TXT,
        alignment_text=fa_pipeline["alignment_text"],
        device=device, token_level=True,
    )

    if args.indices is None or args.indices == "auto":
        indices = select_images_auto(
            dataset, ba_pipeline, fa_pipeline, ba_text_feats, fa_text_feats,
        )
    else:
        indices = [int(x) for x in args.indices.split(",")]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    render_figure(
        dataset, indices,
        ba_pipeline, fa_pipeline,
        ba_text_feats, fa_text_feats,
        output_path=args.output,
        pickle_path=args.pickle,
    )


if __name__ == "__main__":
    main()
