"""Cross-modal anchor attention: shows how the same anchor index
attends to image patches AND text tokens simultaneously.

Demonstrates the "bridge" mechanism — anchor k highlights semantically
matching regions in both modalities.

Two modes:
    per_anchor (default): one row per anchor, col 1 = image heatmap,
        col 2 = text highlighting.
    unified: all anchors overlaid on a single image with distinct colours,
        caption tokens colour-coded by dominant anchor + multi-anchor underlines.

Usage:
    # Original per-anchor layout
    PYTHONPATH=. python scripts/viz/anchor_cross_modal.py \
        --config configs/ba/vitl_roberta/token_k512.yaml \
        --ckpt results/alignment-.../checkpoint-epoch400.pth \
        --layer-img 23 --layer-txt 24 \
        --gpu 0 \
        [--image-idx 42] [--n-anchors 5] \
        [--output drafts/figures/cross_modal_attention.png]

    # Unified multi-anchor overlay
    PYTHONPATH=. python scripts/viz/anchor_cross_modal.py \
        --config configs/ba/vitl_roberta/token_k512.yaml \
        --ckpt results/alignment-.../checkpoint-epoch400.pth \
        --mode unified --n-anchors 5 \
        --output drafts/figures/cross_modal_unified.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.alignment import *  # noqa
from src.core.src.utils.loader import Loader, merge_dicts
from src.models.text.models import load_llm, load_tokenizer
from timm import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.transforms as transforms


def _ensure_rgb(img):
    from PIL import Image as PILImage
    if isinstance(img, PILImage.Image) and img.mode != "RGB":
        return img.convert("RGB")
    return img


def get_image_anchor_attention(image, vision_model, image_transform, alignment_image,
                                layer_img, device):
    """Returns (P, K) attention weights and (P, K) raw similarities."""
    img_t = image_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        lvm_out = vision_model(img_t)
        layer_key = list(lvm_out.keys())[layer_img]
        feats = lvm_out[layer_key].squeeze(0)  # (T, D)

    patches = feats[1:, :].float()  # (P, D), strip CLS
    z_n = F.normalize(patches, dim=-1)
    a_n = F.normalize(alignment_image.anchors.float(), dim=-1)  # (K, D)
    sim = z_n @ a_n.T  # (P, K)

    tau = getattr(alignment_image, 'pool_temperature', 0.05)
    attn = F.softmax(sim / tau, dim=0)  # softmax over patches per anchor

    return attn.detach().cpu().numpy(), sim.detach().cpu().numpy()


def get_text_anchor_attention(text, tokenizer, language_model, alignment_text,
                               layer_txt, device):
    """Returns (T_text, K) attention weights, token strings, raw sims."""
    tokens = tokenizer(text, return_tensors="pt", padding=False, truncation=True)
    input_ids = tokens["input_ids"].to(device)
    attn_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        out = language_model(input_ids=input_ids, attention_mask=attn_mask)
        hidden = torch.stack(out["hidden_states"]).permute(1, 0, 2, 3)  # (B, L, T, D)
        text_feats = hidden[0, layer_txt, :, :]  # (T_text, D)

    z_n = F.normalize(text_feats.float(), dim=-1)
    a_n = F.normalize(alignment_text.anchors.float(), dim=-1)  # (K, D)
    sim = z_n @ a_n.T  # (T_text, K)

    tau = getattr(alignment_text, 'pool_temperature', 0.05)
    # Mask padding
    mask = attn_mask[0].bool()
    logits = sim / tau
    logits[~mask] = float("-inf")
    attn = F.softmax(logits, dim=0)  # softmax over text tokens per anchor
    attn = attn.nan_to_num(0.0)

    # Get token strings
    token_ids = input_ids[0].cpu().tolist()
    token_strs = tokenizer.convert_ids_to_tokens(token_ids)

    return attn.detach().cpu().numpy(), sim.detach().cpu().numpy(), token_strs, mask.cpu().numpy()


def select_top_anchors(img_attn, txt_attn, n=5, strategy="focused"):
    """Select anchors by different strategies.

    Strategies:
        "focused"  — anchors with highest combined focus (low entropy, high max)
        "diverse"  — focused anchors that attend to DIFFERENT image regions
        "consistent" — anchors whose image peak and text peak are most aligned
    """
    K = img_attn.shape[1]
    P = img_attn.shape[0]

    # Pre-compute per-anchor stats
    stats = []
    for k in range(K):
        img_entropy = -np.sum(img_attn[:, k] * np.log(img_attn[:, k] + 1e-10))
        txt_entropy = -np.sum(txt_attn[:, k] * np.log(txt_attn[:, k] + 1e-10))
        img_max = img_attn[:, k].max()
        txt_max = txt_attn[:, k].max()
        img_argmax = img_attn[:, k].argmax()
        txt_argmax = txt_attn[:, k].argmax()
        focus_score = (img_max + txt_max) - 0.3 * (img_entropy + txt_entropy)
        stats.append({
            "k": k, "focus": focus_score,
            "img_max": img_max, "txt_max": txt_max,
            "img_argmax": img_argmax, "txt_argmax": txt_argmax,
            "img_entropy": img_entropy, "txt_entropy": txt_entropy,
            "img_profile": img_attn[:, k],
        })

    if strategy == "focused":
        stats.sort(key=lambda s: -s["focus"])
        return [s["k"] for s in stats[:n]]

    elif strategy == "diverse":
        # Greedy: pick most focused first, then iteratively pick the next
        # most focused anchor whose image attention is most different
        # from all already-selected anchors
        stats.sort(key=lambda s: -s["focus"])
        selected = [stats[0]]
        candidates = stats[1:]

        while len(selected) < n and candidates:
            best_idx, best_score = -1, -1
            for i, cand in enumerate(candidates):
                # Min cosine similarity to any selected anchor's image attention
                min_sim = min(
                    np.dot(cand["img_profile"], sel["img_profile"]) /
                    (np.linalg.norm(cand["img_profile"]) * np.linalg.norm(sel["img_profile"]) + 1e-10)
                    for sel in selected
                )
                # Diversity = low similarity to existing, weighted by focus
                div_score = cand["focus"] * (1.0 - min_sim)
                if div_score > best_score:
                    best_score = div_score
                    best_idx = i

            if best_idx >= 0:
                selected.append(candidates.pop(best_idx))
            else:
                break

        return [s["k"] for s in selected]

    elif strategy == "consistent":
        # Pick anchors where the image-side peak patch and text-side peak
        # token both have high attention AND the anchor is focused
        # Score: geometric mean of img_max and txt_max, penalized by entropy
        for s in stats:
            s["consistency"] = np.sqrt(s["img_max"] * s["txt_max"]) - \
                               0.1 * (s["img_entropy"] + s["txt_entropy"])
        stats.sort(key=lambda s: -s["consistency"])
        return [s["k"] for s in stats[:n]]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def plot_cross_modal(image, caption, img_attn, txt_attn, token_strs,
                     anchor_indices, output_path, img_size=224):
    """Create the cross-modal attention figure."""
    n_anchors = len(anchor_indices)
    P = img_attn.shape[0]
    h = int(np.sqrt(P))

    fig = plt.figure(figsize=(14, 3 * n_anchors + 1))
    gs = gridspec.GridSpec(n_anchors, 2, width_ratios=[1, 1.5],
                           hspace=0.3, wspace=0.05)

    img_np = np.array(image.resize((img_size, img_size)))

    for row, k in enumerate(anchor_indices):
        # Left: image heatmap
        ax_img = fig.add_subplot(gs[row, 0])
        heatmap = img_attn[:, k].reshape(h, h)
        heatmap = np.array(Image.fromarray(heatmap).resize(
            (img_size, img_size), Image.BILINEAR))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        ax_img.imshow(img_np)
        ax_img.imshow(heatmap, cmap='magma', alpha=0.55, vmin=0, vmax=1)
        ax_img.set_title(f"Anchor {k}", fontsize=11, fontweight='bold')
        ax_img.axis('off')

        # Right: text token highlighting
        ax_txt = fig.add_subplot(gs[row, 1])
        ax_txt.axis('off')

        text_weights = txt_attn[:, k]
        # Normalize for color mapping
        tw_norm = (text_weights - text_weights.min()) / (text_weights.max() - text_weights.min() + 1e-8)

        # Build colored text
        x_pos = 0.02
        y_pos = 0.5
        for i, (tok, w) in enumerate(zip(token_strs, tw_norm)):
            # Skip special tokens
            clean = tok.replace("Ġ", " ").replace("▁", " ").replace("##", "")
            if tok in ["<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[PAD]"]:
                continue

            # Color: white (low attention) to red (high attention)
            r = min(1.0, 0.3 + 0.7 * w)
            g = max(0.0, 0.3 - 0.3 * w)
            b = max(0.0, 0.3 - 0.3 * w)
            bg_alpha = 0.1 + 0.6 * w

            txt_obj = ax_txt.text(x_pos, y_pos, clean,
                                   transform=ax_txt.transAxes,
                                   fontsize=12, fontfamily='monospace',
                                   color=(r, g, b),
                                   fontweight='bold' if w > 0.5 else 'normal',
                                   bbox=dict(boxstyle='round,pad=0.15',
                                            facecolor=(1, 0.8, 0.8, bg_alpha),
                                            edgecolor='none') if w > 0.3 else None,
                                   va='center')

            # Get text width for positioning
            fig.canvas.draw()
            try:
                bbox = txt_obj.get_window_extent()
                inv = ax_txt.transAxes.inverted()
                bbox_axes = inv.transform(bbox)
                x_pos = bbox_axes[1][0] + 0.005
            except:
                x_pos += len(clean) * 0.018

            if x_pos > 0.95:
                x_pos = 0.02
                y_pos -= 0.3

    plt.savefig(output_path, bbox_inches='tight', dpi=200, facecolor='white')
    plt.close()
    print(f"Saved to {output_path}")


def _anchor_colors(n):
    """Return n visually distinct RGBA colours for anchor overlays."""
    # Hand-picked for perceptual distinctness on both dark images and white text bg
    palette = [
        (0.90, 0.20, 0.20),  # red
        (0.20, 0.50, 0.90),  # blue
        (0.15, 0.75, 0.35),  # green
        (0.95, 0.65, 0.10),  # orange
        (0.60, 0.25, 0.85),  # purple
        (0.00, 0.75, 0.80),  # teal
        (0.85, 0.45, 0.65),  # pink
        (0.55, 0.55, 0.00),  # olive
    ]
    return palette[:n]


def plot_cross_modal_unified(image, caption, img_attn, txt_attn, token_strs,
                              anchor_indices, output_path, img_size=224):
    """Unified cross-modal figure: all anchors overlaid on one image + one caption.

    Layout:
        Left:  original image with multi-colour anchor heatmaps overlaid
        Right: caption with colour-coded token underlines / highlights
        Bottom: legend mapping colours to anchor indices
    """
    n_anchors = len(anchor_indices)
    P = img_attn.shape[0]
    h = int(np.sqrt(P))
    colors = _anchor_colors(n_anchors)

    fig = plt.figure(figsize=(16, 2.4))
    # Right column: n_anchors text rows stacked vertically beside the image
    gs = gridspec.GridSpec(n_anchors, 2, width_ratios=[0.6, 1.6],
                           hspace=0.0, wspace=-0.05)

    img_np = np.array(image.resize((img_size, img_size))).astype(np.float32) / 255.0

    # ── Left panel: image with multi-anchor heatmap overlay ──────────
    ax_img = fig.add_subplot(gs[:, 0])  # span all rows

    # Build a composite RGB overlay from all anchors
    overlay = np.zeros((*img_np.shape[:2], 3), dtype=np.float32)
    overlay_alpha = np.zeros(img_np.shape[:2], dtype=np.float32)

    for i, k in enumerate(anchor_indices):
        heatmap = img_attn[:, k].reshape(h, h)
        heatmap = np.array(Image.fromarray(heatmap).resize(
            (img_size, img_size), Image.BILINEAR))
        # Normalize per anchor
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        # Threshold: only show top attention regions (above 75th percentile)
        thresh = np.percentile(heatmap, 75)
        mask = heatmap > thresh
        # Smooth falloff above threshold
        intensity = np.where(mask, (heatmap - thresh) / (1.0 - thresh + 1e-8), 0.0)

        for c in range(3):
            overlay[:, :, c] += intensity * colors[i][c]
        overlay_alpha += intensity

    # Normalize overlay to prevent saturation
    overlay_alpha_safe = np.maximum(overlay_alpha, 1e-8)
    for c in range(3):
        overlay[:, :, c] /= overlay_alpha_safe
    overlay_alpha = np.clip(overlay_alpha / overlay_alpha.max(), 0, 0.85)

    # Darken the base image — low-attention regions stay dim, high-attention glows
    # Base brightness: dim everywhere, then restore brightness where anchors attend
    dim_factor = 0.25  # how dark the unattended regions are (0=black, 1=original)
    brightness = dim_factor + (1.0 - dim_factor) * overlay_alpha
    darkened = img_np * brightness[:, :, np.newaxis]

    # Blend coloured overlay on top of the darkened image
    blended = darkened.copy()
    for c in range(3):
        blended[:, :, c] = (
            darkened[:, :, c] * (1.0 - overlay_alpha * 0.6) +
            overlay[:, :, c] * overlay_alpha * 0.6
        )
    blended = np.clip(blended, 0, 1)

    ax_img.imshow(blended)
    ax_img.axis('off')

    # ── Right panel: one text row per anchor ────────────────────────
    # Pre-compute clean tokens (shared across rows)
    clean_tokens = []
    for tok_idx, tok in enumerate(token_strs):
        clean = tok.replace("\u0120", " ").replace("\u2581", " ").replace("##", "")
        if tok in ["<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[PAD]"]:
            continue
        clean_tokens.append((tok_idx, clean))

    # Auto-scale font to fit caption in one line (reserve 0.06 for anchor label)
    total_chars = sum(len(c) for _, c in clean_tokens)
    fontsize = min(15, max(11, int(200 / (total_chars + 1))))

    for row_i, (anchor_k, color) in enumerate(zip(anchor_indices, colors)):
        ax_txt = fig.add_subplot(gs[row_i, 1])
        ax_txt.axis('off')

        tw = txt_attn[:, anchor_k].copy()
        tw = (tw - tw.min()) / (tw.max() - tw.min() + 1e-8)

        r, g, b = color

        ax_txt.text(
            0.0, 0.5, f"A{anchor_k}",
            transform=ax_txt.transAxes,
            fontsize=fontsize - 1, fontfamily='monospace', fontweight='bold',
            color='black', va='center',
        )

        x_pos = 0.06

        for tok_idx, clean in clean_tokens:
            w = tw[tok_idx]
            if w > 0.3:
                bg_alpha = 0.15 + 0.6 * w
                bbox_props = dict(
                    boxstyle='round,pad=0.12',
                    facecolor=(r, g, b, bg_alpha),
                    edgecolor='none',
                )
                fontweight = 'bold' if w > 0.5 else 'normal'
                text_color = (r * 0.6, g * 0.6, b * 0.6)
            else:
                bbox_props = None
                fontweight = 'normal'
                text_color = (0.4, 0.4, 0.4)

            txt_obj = ax_txt.text(
                x_pos, 0.5, clean,
                transform=ax_txt.transAxes,
                fontsize=fontsize, fontfamily='monospace',
                color=text_color,
                fontweight=fontweight,
                bbox=bbox_props,
                va='center',
            )

            fig.canvas.draw()
            try:
                bbox = txt_obj.get_window_extent()
                inv = ax_txt.transAxes.inverted()
                bbox_axes = inv.transform(bbox)
                x_pos = bbox_axes[1][0] + 0.003
            except Exception:
                x_pos += len(clean) * 0.014

    plt.savefig(output_path, bbox_inches='tight', dpi=200, facecolor='white')
    plt.close()
    print(f"Saved unified figure to {output_path}")


def plot_cross_modal_grid(samples, output_path, img_size=224, bar_orientation="horizontal"):
    """Grid layout: rows = samples, cols = [original, top-1, top-2, top-3 anchors].

    Each anchor column has two sub-panels: image heatmap + token attention bar plot.

    Parameters
    ----------
    samples : list of dict, each with keys:
        "image": PIL.Image, "caption": str,
        "img_attn": (P, K), "txt_attn": (T, K),
        "token_strs": list[str], "anchor_indices": list[int] (top-3)
    bar_orientation : "horizontal" or "vertical"
    """
    n_rows = len(samples)
    n_anchor_cols = 3  # top-1, top-2, top-3

    col_width = 9.0 / 4
    img_height = col_width

    fig = plt.figure(figsize=(9, img_height * n_rows * 1.25))
    gs = gridspec.GridSpec(n_rows, 4,
                           width_ratios=[1, 1, 1, 1],
                           hspace=0.35, wspace=0.08)

    fig.patch.set_facecolor("white")

    for row_i, sample in enumerate(samples):
        pil_img = sample["image"]
        img_attn = sample["img_attn"]
        txt_attn = sample["txt_attn"]
        token_strs = sample["token_strs"]
        anchor_indices = sample["anchor_indices"][:n_anchor_cols]

        P = img_attn.shape[0]
        h = int(np.sqrt(P))
        img_np = np.array(pil_img.resize((img_size, img_size)))

        # -- Column 0: original image with caption --
        ax_orig = fig.add_subplot(gs[row_i, 0])
        ax_orig.imshow(img_np)
        ax_orig.axis("off")
        from textwrap import fill, wrap
        # Estimate chars per line: image is ~col_width inches at fontsize 6
        chars_per_line = int(col_width * 15.5)  # ~15.5 monospace chars per inch at size 6

        # Build word list with zero attention (same rendering as cols 2-4)
        col0_words = []
        for tok_idx, tok in enumerate(token_strs):
            clean = tok.replace("\u0120", " ").replace("\u2581", " ").replace("##", "").strip()
            if tok in ["<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[PAD]"]:
                continue
            if clean:
                col0_words.append((clean, 0.0))  # all zero attention

        caption_str = " ".join(w for w, _ in col0_words)
        lines_col0 = wrap(caption_str, width=chars_per_line)

        ax_cap0 = ax_orig.inset_axes([0.0, -0.50, 1.0, 0.45])
        ax_cap0.set_xlim(0, 1)
        ax_cap0.set_ylim(0, 1)
        ax_cap0.axis("off")

        cap_fontsize = 6
        line_height = 0.17
        word_idx_0 = 0

        for li, line in enumerate(lines_col0):
            line_words = line.split(" ")
            tmp = ax_cap0.text(0.5, 0.95 - li * line_height, line,
                               transform=ax_cap0.transAxes,
                               fontsize=cap_fontsize, fontfamily="monospace",
                               color="none", ha="center", va="center")
            fig.canvas.draw()
            try:
                bb = tmp.get_window_extent()
                inv = ax_cap0.transAxes.inverted()
                bb_ax = inv.transform(bb)
                line_start_x = bb_ax[0][0]
            except Exception:
                line_start_x = 0.05
            tmp.remove()

            x_pos = line_start_x
            y_pos = 0.95 - li * line_height

            for wi, word in enumerate(line_words):
                word_idx_0 += 1
                display_word = word + " "
                txt_obj = ax_cap0.text(
                    x_pos, y_pos, display_word,
                    transform=ax_cap0.transAxes,
                    fontsize=cap_fontsize, fontfamily="monospace",
                    color=(0.3, 0.3, 0.3),
                    fontweight="normal",
                    va="center", ha="left",
                )
                fig.canvas.draw()
                try:
                    bb = txt_obj.get_window_extent()
                    inv = ax_cap0.transAxes.inverted()
                    bb_ax = inv.transform(bb)
                    x_pos = bb_ax[1][0]
                except Exception:
                    x_pos += len(display_word) * 0.022

        # -- Columns 1-3: anchor heatmap with token labels --
        for col_i, k in enumerate(anchor_indices):
            heatmap = img_attn[:, k].reshape(h, h)
            heatmap = np.array(Image.fromarray(heatmap).resize(
                (img_size, img_size), Image.BILINEAR))
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

            ax_hm = fig.add_subplot(gs[row_i, 1 + col_i])
            ax_hm.imshow(img_np)
            ax_hm.imshow(heatmap, cmap="magma", alpha=0.55, vmin=0, vmax=1)
            ax_hm.set_title(f"Anchor {k}", fontsize=9, fontweight="bold")
            ax_hm.axis("off")

            # Caption below heatmap — word-by-word rendering with highlighting
            # Same visual result as plot_cross_modal: gradient color + pink bbox
            text_weights = txt_attn[:, k].copy()
            tw_norm = (text_weights - text_weights.min()) / \
                      (text_weights.max() - text_weights.min() + 1e-8)

            # Build word list with attention weights
            cap_words = []
            for tok_idx, tok in enumerate(token_strs):
                clean = tok.replace("\u0120", " ").replace("\u2581", " ").replace("##", "").strip()
                if tok in ["<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[PAD]"]:
                    continue
                if clean:
                    cap_words.append((clean, tw_norm[tok_idx]))

            # Use textwrap to determine line breaks
            from textwrap import wrap
            caption_str = " ".join(w for w, _ in cap_words)
            lines = wrap(caption_str, width=chars_per_line)

            # Create text axes below heatmap
            ax_cap = ax_hm.inset_axes([0.0, -0.50, 1.0, 0.45])
            ax_cap.set_xlim(0, 1)
            ax_cap.set_ylim(0, 1)
            ax_cap.axis("off")

            cap_fontsize = 6
            line_height = 0.17
            word_idx = 0

            for li, line in enumerate(lines):
                line_words = line.split(" ")
                # Center each line: measure total width first
                # Render a hidden version to get total line width
                tmp = ax_cap.text(0.5, 0.95 - li * line_height, line,
                                  transform=ax_cap.transAxes,
                                  fontsize=cap_fontsize, fontfamily="monospace",
                                  color="none", ha="center", va="center")
                fig.canvas.draw()
                try:
                    bb = tmp.get_window_extent()
                    inv = ax_cap.transAxes.inverted()
                    bb_ax = inv.transform(bb)
                    line_start_x = bb_ax[0][0]
                except Exception:
                    line_start_x = 0.05
                tmp.remove()

                # Now render word by word from line_start_x
                x_pos = line_start_x
                y_pos = 0.95 - li * line_height

                for wi, word in enumerate(line_words):
                    if word_idx < len(cap_words):
                        _, w = cap_words[word_idx]
                    else:
                        w = 0
                    word_idx += 1

                    # Color: match column 1 grey for non-highlighted,
                    # gradient to red only for highlighted tokens
                    if w > 0.3:
                        r = min(1.0, 0.3 + 0.7 * w)
                        g_c = max(0.0, 0.3 - 0.3 * w)
                        b_c = max(0.0, 0.3 - 0.3 * w)
                        bg_alpha = 0.1 + 0.6 * w
                        text_color = (r, g_c, b_c)
                        bbox_props = dict(boxstyle="round,pad=0.1",
                                          facecolor=(1, 0.8, 0.8, bg_alpha),
                                          edgecolor="none")
                    else:
                        text_color = (0.3, 0.3, 0.3)
                        bbox_props = None

                    if bbox_props is not None:
                        # Render word without trailing space so bbox fits tight
                        txt_obj = ax_cap.text(
                            x_pos, y_pos, word,
                            transform=ax_cap.transAxes,
                            fontsize=cap_fontsize, fontfamily="monospace",
                            color=text_color,
                            fontweight="bold" if w > 0.5 else "normal",
                            bbox=bbox_props,
                            va="center", ha="left",
                        )
                        fig.canvas.draw()
                        try:
                            bb = txt_obj.get_window_extent()
                            inv = ax_cap.transAxes.inverted()
                            bb_ax = inv.transform(bb)
                            x_pos = bb_ax[1][0]
                        except Exception:
                            x_pos += len(word) * 0.022
                        # Add trailing space as separate unboxed text
                        sp_obj = ax_cap.text(
                            x_pos, y_pos, " ",
                            transform=ax_cap.transAxes,
                            fontsize=cap_fontsize, fontfamily="monospace",
                            color=text_color, va="center", ha="left",
                        )
                        fig.canvas.draw()
                        try:
                            bb = sp_obj.get_window_extent()
                            inv = ax_cap.transAxes.inverted()
                            bb_ax = inv.transform(bb)
                            x_pos = bb_ax[1][0]
                        except Exception:
                            x_pos += 0.015
                    else:
                        display_word = word + " "
                        txt_obj = ax_cap.text(
                            x_pos, y_pos, display_word,
                            transform=ax_cap.transAxes,
                            fontsize=cap_fontsize, fontfamily="monospace",
                            color=text_color,
                            fontweight="bold" if w > 0.5 else "normal",
                            va="center", ha="left",
                        )
                        fig.canvas.draw()
                        try:
                            bb = txt_obj.get_window_extent()
                            inv = ax_cap.transAxes.inverted()
                            bb_ax = inv.transform(bb)
                            x_pos = bb_ax[1][0]
                        except Exception:
                            x_pos += len(display_word) * 0.022

    # Clean up all axes: remove ticks, borders
    for ax in fig.axes:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.savefig(output_path, bbox_inches="tight", dpi=200,
                facecolor="white", edgecolor="none")
    pdf_path = str(output_path).rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved grid figure to {output_path} and {pdf_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--layer-img", type=int, default=23)
    p.add_argument("--layer-txt", type=int, default=24)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--image-idx", type=int, default=42,
                   help="Dataset sample index")
    p.add_argument("--dataset", choices=["coco", "flickr30"], default="coco",
                   help="Which dataset to pull the image+caption from")
    p.add_argument("--caption", default=None,
                   help="Override caption (otherwise uses dataset annotation)")
    p.add_argument("--n-anchors", type=int, default=5)
    p.add_argument("--mode", choices=["per_anchor", "unified", "grid"], default="per_anchor",
                   help="per_anchor: one row per anchor (original). "
                        "unified: all anchors on one image+caption. "
                        "grid: multi-sample grid with heatmaps + bar plots.")
    p.add_argument("--image-indices", default=None,
                   help="Comma-separated indices for grid mode (e.g. '0,5')")
    p.add_argument("--anchor-overrides", default=None,
                   help="Per-sample anchor overrides for grid mode. "
                        "Semi-colon separated per sample, comma separated per anchor. "
                        "e.g. '280,237,106;341,490,237' overrides both samples.")
    p.add_argument("--bar-orientation", choices=["horizontal", "vertical"],
                   default="horizontal",
                   help="Bar plot orientation in grid mode")
    p.add_argument("--output", default="drafts/figures/cross_modal_attention.png")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")

    # Load config
    with open(args.config) as f:
        cfg = Loader(f).get_single_data()
    cfg = merge_dicts(cfg.get("defaults", {}), cfg.get("overrides", {}))

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    alignment_image = ckpt["alignment_image"].eval().to(device)
    alignment_text = ckpt["alignment_text"].eval().to(device)

    # Load encoders
    lvm_name = cfg["alignment"]["lvm_model_name"]
    llm_name = cfg["alignment"]["llm_model_name"]
    img_size = int(cfg["features"].get("img_size", 224))

    vision_model = create_model(lvm_name, pretrained=True, img_size=img_size)
    data_config = resolve_data_config(vision_model.pretrained_cfg, model=vision_model)
    data_config["input_size"] = (3, img_size, img_size)
    data_config["crop_pct"] = 1.0
    return_nodes = [f"blocks.{i}.add_1" for i in range(len(vision_model.blocks))]
    vision_model = create_feature_extractor(vision_model, return_nodes=return_nodes)
    vision_model = vision_model.eval().to(device)

    image_transform = transforms.Compose([
        transforms.Lambda(_ensure_rgb),
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config["mean"], std=data_config["std"]),
    ])

    language_model = load_llm(llm_name).to(device)
    tokenizer = load_tokenizer(llm_name)

    # Helper: load image + caption for a given index
    import csv, json

    def load_sample(idx):
        if args.dataset == "flickr30":
            with open("data/flickr30k/test.txt") as f:
                test_ids = [line.strip() for line in f if line.strip()]
            img_id = test_ids[idx]
            img_path = f"data/flickr30k/images/{img_id}.jpg"
            caption = "a photo"
            with open("data/flickr30k/results.csv") as f:
                reader = csv.reader(f, delimiter="|")
                for row in reader:
                    if row[0] == f"{img_id}.jpg" and row[1].strip() == "0":
                        caption = row[2].strip()
                        break
        else:
            with open("data/COCO/annotations/captions_val2014.json") as f:
                coco_data = json.load(f)
            img_info = coco_data["images"][idx]
            img_id = img_info["id"]
            img_path = f"data/COCO/val2014/{img_info['file_name']}"
            caps = [a["caption"] for a in coco_data["annotations"] if a["image_id"] == img_id]
            caption = caps[0] if caps else "a photo"
        return Image.open(img_path).convert("RGB"), caption, img_path

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "grid":
        # Multi-sample grid
        indices = [int(x) for x in args.image_indices.split(",")]
        # Parse anchor overrides if provided
        anchor_override_list = None
        if args.anchor_overrides:
            anchor_override_list = [
                [int(a) for a in s.split(",")]
                for s in args.anchor_overrides.split(";")
            ]
        samples = []
        for si, idx in enumerate(indices):
            pil_img, caption, img_path = load_sample(idx)
            print(f"Image [{idx}]: {img_path}")
            print(f"Caption [{idx}]: {caption}")
            img_attn, _ = get_image_anchor_attention(
                pil_img, vision_model, image_transform, alignment_image,
                args.layer_img, device)
            txt_attn, _, token_strs, _ = get_text_anchor_attention(
                caption, tokenizer, language_model, alignment_text,
                args.layer_txt, device)
            if anchor_override_list and si < len(anchor_override_list):
                anchor_indices = anchor_override_list[si]
            else:
                anchor_indices = select_top_anchors(img_attn, txt_attn, n=3,
                                                    strategy="focused")
            print(f"  Anchors: {anchor_indices}")
            samples.append({
                "image": pil_img, "caption": caption,
                "img_attn": img_attn, "txt_attn": txt_attn,
                "token_strs": token_strs, "anchor_indices": anchor_indices,
            })
        plot_cross_modal_grid(samples, args.output, img_size,
                              bar_orientation=args.bar_orientation)
    else:
        # Single-sample modes
        pil_img, caption, img_path = load_sample(args.image_idx)
        if args.caption:
            caption = args.caption
        print(f"Image: {img_path}")
        print(f"Caption: {caption}")

        img_attn, img_sim = get_image_anchor_attention(
            pil_img, vision_model, image_transform, alignment_image,
            args.layer_img, device)
        txt_attn, txt_sim, token_strs, txt_mask = get_text_anchor_attention(
            caption, tokenizer, language_model, alignment_text,
            args.layer_txt, device)
        anchor_indices = select_top_anchors(img_attn, txt_attn, n=args.n_anchors)
        print(f"Selected anchors: {anchor_indices}")

        if args.mode == "unified":
            plot_cross_modal_unified(pil_img, caption, img_attn, txt_attn, token_strs,
                                     anchor_indices, args.output, img_size)
        else:
            plot_cross_modal(pil_img, caption, img_attn, txt_attn, token_strs,
                             anchor_indices, args.output, img_size)


if __name__ == "__main__":
    main()
