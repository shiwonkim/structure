"""Shared visualization utilities for paper figures.

Provides model loading, single-image segmentation inference, mask overlay,
and the Pascal VOC standard colour palette.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from src.core.src.utils.loader import Loader, merge_dicts
from src.evaluation.zero_shot_segmentation import (
    DatasetSpec,
    build_language_encoder,
    build_vision_encoder,
    get_text_templates,
)
from src.evaluation.zero_shot_classifier import build_zero_shot_classifier


# ------------------------------------------------------------------
# Config + checkpoint loading
# ------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        raw = yaml.load(f, Loader=Loader)
    return merge_dicts(raw.get("defaults", {}), raw.get("overrides", {}))


def load_checkpoint(
    checkpoint_path: str, device: torch.device
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    alignment_image = ckpt["alignment_image"].to(device).eval()
    alignment_text = ckpt["alignment_text"].to(device).eval()
    if hasattr(alignment_image, "set_modality"):
        alignment_image.set_modality("image")
    if hasattr(alignment_text, "set_modality"):
        alignment_text.set_modality("text")
    return alignment_image, alignment_text


def load_full_pipeline(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> dict:
    """Load config, encoders, and alignment layers in one call."""
    cfg = load_config(config_path)
    vision_model, image_transform, img_size = build_vision_encoder(cfg, device)
    language_model, tokenizer = build_language_encoder(cfg, device)
    alignment_image, alignment_text = load_checkpoint(checkpoint_path, device)
    return {
        "cfg": cfg,
        "vision_model": vision_model,
        "image_transform": image_transform,
        "img_size": img_size,
        "language_model": language_model,
        "tokenizer": tokenizer,
        "alignment_image": alignment_image,
        "alignment_text": alignment_text,
        "device": device,
    }


# ------------------------------------------------------------------
# Vision forward — extract per-layer features for a single PIL image
# ------------------------------------------------------------------

def extract_patch_features(
    pil_img: Image.Image,
    vision_model: torch.nn.Module,
    image_transform,
    layer_img: int,
    device: torch.device,
) -> torch.Tensor:
    """Return (T, D) features for the selected layer (T = 1 CLS + P patches)."""
    img_t = image_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        lvm_out = vision_model(img_t)
        layer_key = list(lvm_out.keys())[layer_img]
        return lvm_out[layer_key].squeeze(0)  # (T, D)


# ------------------------------------------------------------------
# Text features
# ------------------------------------------------------------------

def build_text_features(
    classnames: Sequence[str],
    strategy: str,
    tokenizer,
    language_model: torch.nn.Module,
    layer_txt: int,
    alignment_text: Optional[torch.nn.Module],
    device: torch.device,
    token_level: bool = True,
) -> torch.Tensor:
    """Build (C, D_method) class descriptors via the ZS classifier builder."""
    templates = get_text_templates(strategy)
    pool_txt = "none" if token_level else "avg"
    return build_zero_shot_classifier(
        language_model=language_model,
        tokenizer=tokenizer,
        classnames=classnames,
        templates=templates,
        dataset=None,
        layer_index=layer_txt,
        alignment_layer=alignment_text,
        num_classes_per_batch=8,
        device=device,
        pool_txt=pool_txt,
        save_path=None,
        token_level=token_level,
    ).to(device)


# ------------------------------------------------------------------
# Single-image segmentation
# ------------------------------------------------------------------

def segment_image_ba_codebook(
    layer_feats: torch.Tensor,
    text_feats: torch.Tensor,
    alignment_image: torch.nn.Module,
    gt_shape: Tuple[int, int],
    device: torch.device,
) -> np.ndarray:
    """Anchor-codebook segmentation for one image. Returns (H, W) int prediction."""
    with torch.no_grad():
        patches = layer_feats[1:, :].to(device)  # (P, D)
        z_n = F.normalize(patches, dim=-1)
        a_n = F.normalize(alignment_image.anchors, dim=-1)
        patch_feats = z_n @ a_n.T  # (P, K)
    return _sim_to_pred(patch_feats, text_feats, gt_shape)


def segment_image_freezealign(
    layer_feats: torch.Tensor,
    text_feats: torch.Tensor,
    alignment_image: torch.nn.Module,
    gt_shape: Tuple[int, int],
    device: torch.device,
) -> np.ndarray:
    """FreezeAlign segmentation for one image."""
    with torch.no_grad():
        z = layer_feats.unsqueeze(0).to(device)
        projected = alignment_image.local_vision_proj(z)
        patch_feats = projected.squeeze(0)[1:, :]  # (P, E)
    return _sim_to_pred(patch_feats, text_feats, gt_shape)


def _sim_to_pred(
    patch_feats: torch.Tensor,
    text_feats: torch.Tensor,
    gt_shape: Tuple[int, int],
) -> np.ndarray:
    """Shared path: cosine sim → bilinear upsample → argmax → (H, W)."""
    patch_feats = F.normalize(patch_feats, dim=-1)
    text_feats_n = F.normalize(text_feats, dim=-1)
    sim = patch_feats @ text_feats_n.T  # (P, C)
    P = sim.shape[0]
    h = int(round(math.sqrt(P)))
    C = sim.shape[1]
    sim_map = sim.view(h, h, C).permute(2, 0, 1).unsqueeze(0)  # (1, C, h, h)
    H, W = gt_shape
    sim_up = F.interpolate(
        sim_map.float(), size=(H, W), mode="bilinear", align_corners=False,
    )
    return sim_up.squeeze(0).argmax(dim=0).cpu().numpy().astype(np.int64)


# ------------------------------------------------------------------
# Per-anchor CAP attention (for heatmap figure)
# ------------------------------------------------------------------

def compute_anchor_attention(
    layer_feats: torch.Tensor,
    alignment_image: torch.nn.Module,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    """Compute per-anchor attention over patches.

    Returns
    -------
    attn : (P, K) float tensor — softmax(sim / tau, dim=0) over patches.
    h    : int — patch grid side length (h = w = sqrt(P)).
    """
    with torch.no_grad():
        patches = layer_feats[1:, :].to(device)  # (P, D), strip CLS
        z_n = F.normalize(patches, dim=-1)
        a_n = F.normalize(alignment_image.anchors, dim=-1)  # (K, D)
        sim = z_n @ a_n.T  # (P, K)
        tau = alignment_image.pool_temperature
        attn = F.softmax(sim / tau, dim=0)  # softmax over patches
    P = attn.shape[0]
    h = int(round(math.sqrt(P)))
    return attn, h


def anchor_class_correspondence(
    text_feats: torch.Tensor,
    classnames: Sequence[str],
) -> List[Tuple[int, str, float]]:
    """For each anchor k, find the class with the highest text profile value.

    Parameters
    ----------
    text_feats : (C, K) — L2-normalised per-class CAP profiles.
    classnames : length-C list of class display names.

    Returns
    -------
    List of (anchor_index, class_name, activation) for all K anchors.
    """
    # text_feats: (C, K).  For each anchor k, find argmax over C.
    if text_feats.dim() != 2:
        raise ValueError(f"Expected (C, K) text features, got {text_feats.shape}")
    K = text_feats.shape[1]
    out = []
    for k in range(K):
        col = text_feats[:, k]  # (C,)
        idx = col.argmax().item()
        out.append((k, classnames[idx], col[idx].item()))
    return out


# ------------------------------------------------------------------
# Pascal VOC colour palette
# ------------------------------------------------------------------

def pascal_voc_palette() -> np.ndarray:
    """Standard Pascal VOC 2012 colour palette, shape (256, 3) uint8.

    Index 0 = background (black), 1..20 = foreground classes,
    255 = ignore (white, but typically masked out).
    """
    palette = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        palette[i] = [r, g, b]
    return palette


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.6,
    palette: Optional[np.ndarray] = None,
    ignore_index: int = 255,
) -> np.ndarray:
    """Alpha-blend a class-index mask onto an RGB image.

    Parameters
    ----------
    image   : (H, W, 3) uint8 RGB.
    mask    : (H, W) int — class indices.
    alpha   : blend strength of the colour overlay.
    palette : (N, 3) uint8 colour map; defaults to Pascal VOC palette.

    Returns
    -------
    (H, W, 3) uint8 RGB with overlay.
    """
    if palette is None:
        palette = pascal_voc_palette()
    colour = palette[np.clip(mask, 0, len(palette) - 1)]  # (H, W, 3)
    valid = mask != ignore_index
    out = image.copy()
    out[valid] = (
        (1 - alpha) * image[valid].astype(np.float32)
        + alpha * colour[valid].astype(np.float32)
    ).clip(0, 255).astype(np.uint8)
    return out


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    cmap: str = "viridis",
) -> np.ndarray:
    """Alpha-blend a scalar heatmap onto an RGB image using a matplotlib cmap.

    Parameters
    ----------
    image   : (H, W, 3) uint8 RGB.
    heatmap : (H, W) float in [0, 1].
    alpha   : blend strength.
    cmap    : matplotlib colormap name.

    Returns
    -------
    (H, W, 3) uint8 RGB with heatmap overlay.
    """
    import matplotlib.cm as cm
    mapper = cm.get_cmap(cmap)
    coloured = (mapper(heatmap)[:, :, :3] * 255).astype(np.uint8)  # (H, W, 3)
    blended = (
        (1 - alpha) * image.astype(np.float32)
        + alpha * coloured.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)
    return blended
