"""Cross-modal anchor consistency: continuous weighted overlap and cosine
similarity for matched vs mismatched image-text pairs.

Metrics:
  1. Top-k truncated weighted Jaccard: soft anchor support overlap
  2. Top-k truncated histogram intersection (optional comparison)
  3. Profile cosine similarity
  4. Discrete top-k overlap ratio (for summary comparison only)

Usage:
    PYTHONPATH=. python scripts/viz/anchor_cross_modal_consistency.py \
        --load-cache results/anchor_stats_flickr_cache.pkl \
        --output drafts/figures/anchor_consistency.png

Runtime: <2s from cache for 1000 samples × 5 negatives.
"""
import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


# ── Metrics ──────────────────────────────────────────────────────────

def _truncate_topk(profile, k):
    """Zero out all but the top-k entries. Returns a copy."""
    out = np.zeros_like(profile)
    topk_idx = np.argsort(-profile)[:k]
    out[topk_idx] = profile[topk_idx]
    return out


def weighted_jaccard(profile_a, profile_b, k):
    """Top-k truncated weighted Jaccard similarity.

    1. Keep only top-k entries in each profile, zero rest.
    2. WJ = sum min(a,b) / sum max(a,b)
    Returns float in [0, 1].
    """
    a = _truncate_topk(profile_a, k)
    b = _truncate_topk(profile_b, k)
    num = np.minimum(a, b).sum()
    den = np.maximum(a, b).sum()
    if den < 1e-12:
        return 0.0
    return float(num / den)


def histogram_intersection(profile_a, profile_b, k):
    """Top-k truncated histogram intersection after L1 normalization.

    HI = sum min(a_norm, b_norm)
    Returns float in [0, 1].
    """
    a = _truncate_topk(profile_a, k)
    b = _truncate_topk(profile_b, k)
    a_sum = a.sum()
    b_sum = b.sum()
    if a_sum < 1e-12 or b_sum < 1e-12:
        return 0.0
    a_norm = a / a_sum
    b_norm = b / b_sum
    return float(np.minimum(a_norm, b_norm).sum())


def topk_overlap_ratio(profile_a, profile_b, k):
    """Discrete top-k set overlap: |topk(a) ∩ topk(b)| / k."""
    top_a = set(np.argsort(-profile_a)[:k])
    top_b = set(np.argsort(-profile_b)[:k])
    return len(top_a & top_b) / k


def cosine_similarity(a, b):
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ── Computation ──────────────────────────────────────────────────────

def compute_consistency(image_profiles, text_profiles, top_k=5,
                         num_negatives=5, seed=42):
    """Compute all metrics for matched and mismatched pairs."""
    rng = np.random.RandomState(seed)
    N = image_profiles.shape[0]

    matched = {
        "wj": np.zeros(N), "hi": np.zeros(N),
        "discrete": np.zeros(N), "cosine": np.zeros(N),
    }
    for i in range(N):
        pi, pt = image_profiles[i], text_profiles[i]
        matched["wj"][i] = weighted_jaccard(pi, pt, top_k)
        matched["hi"][i] = histogram_intersection(pi, pt, top_k)
        matched["discrete"][i] = topk_overlap_ratio(pi, pt, top_k)
        matched["cosine"][i] = cosine_similarity(pi, pt)

    n_mis = N * num_negatives
    mismatched = {
        "wj": np.zeros(n_mis), "hi": np.zeros(n_mis),
        "discrete": np.zeros(n_mis), "cosine": np.zeros(n_mis),
    }
    idx = 0
    for i in range(N):
        candidates = np.delete(np.arange(N), i)
        neg_indices = rng.choice(candidates, size=num_negatives, replace=False)
        for j in neg_indices:
            pi, pt = image_profiles[i], text_profiles[j]
            mismatched["wj"][idx] = weighted_jaccard(pi, pt, top_k)
            mismatched["hi"][idx] = histogram_intersection(pi, pt, top_k)
            mismatched["discrete"][idx] = topk_overlap_ratio(pi, pt, top_k)
            mismatched["cosine"][idx] = cosine_similarity(pi, pt)
            idx += 1

    return matched, mismatched


# ── Plotting ─────────────────────────────────────────────────────────

def _kde_curve(data, x_grid, bw_method="scott"):
    """KDE density estimate."""
    try:
        kde = gaussian_kde(data, bw_method=bw_method)
        return kde(x_grid)
    except Exception:
        counts, edges = np.histogram(data, bins=50, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return np.interp(x_grid, centers, counts)


def plot_consistency(matched, mismatched, output_path, top_k):
    """Two-panel figure: weighted Jaccard KDE + cosine similarity KDE."""
    c_match = "#264653"    # dark teal
    c_mismatch = "#E76F51"  # burnt sienna

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.5, 3.4))
    fig.patch.set_facecolor("white")

    # ── Left panel: weighted Jaccard ──
    all_wj = np.concatenate([matched["wj"], mismatched["wj"]])
    x_wj = np.linspace(max(0, all_wj.min() - 0.02),
                        min(1, all_wj.max() + 0.05), 300)

    kde_m = _kde_curve(matched["wj"], x_wj)
    kde_mis = _kde_curve(mismatched["wj"], x_wj)

    ax_left.fill_between(x_wj, kde_m, alpha=0.35, color=c_match)
    ax_left.plot(x_wj, kde_m, color=c_match, linewidth=2, label="Matched")
    ax_left.fill_between(x_wj, kde_mis, alpha=0.35, color=c_mismatch)
    ax_left.plot(x_wj, kde_mis, color=c_mismatch, linewidth=2, label="Mismatched")

    ax_left.axvline(matched["wj"].mean(), color=c_match,
                    linestyle="--", linewidth=1.2, alpha=0.7)
    ax_left.axvline(mismatched["wj"].mean(), color=c_mismatch,
                    linestyle="--", linewidth=1.2, alpha=0.7)

    ax_left.set_xlabel("Weighted anchor overlap", fontsize=12)
    ax_left.set_ylabel("Density", fontsize=12)
    ax_left.tick_params(labelsize=10)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.set_facecolor("white")
    ax_left.legend(fontsize=10, loc="upper right", framealpha=1.0,
                   edgecolor="#cccccc")

    # ── Right panel: cosine similarity ──
    all_cos = np.concatenate([matched["cosine"], mismatched["cosine"]])
    x_cos = np.linspace(max(0, all_cos.min() - 0.05),
                         min(1, all_cos.max() + 0.05), 300)

    kde_m_cos = _kde_curve(matched["cosine"], x_cos)
    kde_mis_cos = _kde_curve(mismatched["cosine"], x_cos)

    ax_right.fill_between(x_cos, kde_m_cos, alpha=0.35, color=c_match)
    ax_right.plot(x_cos, kde_m_cos, color=c_match, linewidth=2, label="Matched")
    ax_right.fill_between(x_cos, kde_mis_cos, alpha=0.35, color=c_mismatch)
    ax_right.plot(x_cos, kde_mis_cos, color=c_mismatch, linewidth=2, label="Mismatched")

    ax_right.axvline(matched["cosine"].mean(), color=c_match,
                     linestyle="--", linewidth=1.2, alpha=0.7)
    ax_right.axvline(mismatched["cosine"].mean(), color=c_mismatch,
                     linestyle="--", linewidth=1.2, alpha=0.7)

    ax_right.set_xlabel("Profile cosine similarity", fontsize=12)
    ax_right.set_ylabel("Density", fontsize=12)
    ax_right.tick_params(labelsize=10)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.set_facecolor("white")
    ax_right.legend(fontsize=10, loc="upper left", framealpha=1.0,
                    edgecolor="#cccccc")

    fig.tight_layout(pad=0.5)

    fig.savefig(output_path, bbox_inches="tight", dpi=300,
                facecolor="white", edgecolor="none")
    pdf_path = str(output_path).rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300,
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved to {output_path} and {pdf_path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load-cache", required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--num-negatives", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="drafts/figures/anchor_consistency.png")
    args = p.parse_args()

    print(f"Loading cache from {args.load_cache}")
    with open(args.load_cache, "rb") as f:
        cache = pickle.load(f)
    image_profiles = cache["image_profiles"]
    text_profiles = cache["text_profiles"]
    N, K = image_profiles.shape
    print(f"Loaded {N} samples, K={K}")

    matched, mismatched = compute_consistency(
        image_profiles, text_profiles,
        top_k=args.top_k, num_negatives=args.num_negatives, seed=args.seed)

    # ── Summary ──
    n_match = len(matched["wj"])
    n_mis = len(mismatched["wj"])

    print(f"\n{'=' * 72}")
    print(f"CROSS-MODAL ANCHOR CONSISTENCY  (top-{args.top_k}, "
          f"{args.num_negatives} neg/sample)")
    print(f"{'=' * 72}")
    print(f"  {'Metric':30s}  {'Mean':>8s}  {'Median':>8s}  {'Std':>8s}  {'N':>7s}")
    print(f"  {'-' * 67}")

    metrics = [
        ("Weighted Jaccard (matched)", matched["wj"]),
        ("Weighted Jaccard (mismatched)", mismatched["wj"]),
        ("Hist. intersection (matched)", matched["hi"]),
        ("Hist. intersection (mismatched)", mismatched["hi"]),
        ("Discrete overlap (matched)", matched["discrete"]),
        ("Discrete overlap (mismatched)", mismatched["discrete"]),
        ("Cosine sim. (matched)", matched["cosine"]),
        ("Cosine sim. (mismatched)", mismatched["cosine"]),
    ]
    for label, d in metrics:
        print(f"  {label:30s}  {d.mean():8.4f}  {np.median(d):8.4f}  "
              f"{d.std():8.4f}  {len(d):>7d}")

    print(f"\n  {'Metric':30s}  {'Gap (match - mismatch)':>22s}")
    print(f"  {'-' * 55}")
    for name, mk, msk in [
        ("Weighted Jaccard", matched["wj"], mismatched["wj"]),
        ("Hist. intersection", matched["hi"], mismatched["hi"]),
        ("Discrete top-k overlap", matched["discrete"], mismatched["discrete"]),
        ("Cosine similarity", matched["cosine"], mismatched["cosine"]),
    ]:
        gap = mk.mean() - msk.mean()
        ratio = mk.mean() / max(msk.mean(), 1e-12)
        print(f"  {name:30s}  {gap:+8.4f}  ({ratio:.2f}x)")

    print(f"{'=' * 72}")

    # Plot
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_consistency(matched, mismatched, args.output, args.top_k)


if __name__ == "__main__":
    main()
