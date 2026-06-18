"""K-sweep ablation plot for narrow NeurIPS wrapfigure/minipage layout.

Effect of anchor count K on BA performance: avg classification and
avg retrieval vs K, with horizontal dashed baselines.

Usage:
    PYTHONPATH=. python scripts/viz/plot_k_sweep.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Results ──────────────────────────────────────────────────────────
BA_RESULTS = {
    32: {
        "cls": {"stl10": 89.4, "cifar10": 91.3, "cifar100": 40.1,
                "imagenet": 13.6, "eurosat": 28.1},
        "ret": {"flickr_i2t": 58.7, "flickr_t2i": 46.8,
                "coco_i2t": 40.5, "coco_t2i": 30.4},
    },
    64: {
        "cls": {"stl10": 91.4, "cifar10": 93.5, "cifar100": 45.0,
                "imagenet": 16.9, "eurosat": 33.9},
        "ret": {"flickr_i2t": 68.1, "flickr_t2i": 52.0,
                "coco_i2t": 45.5, "coco_t2i": 34.3},
    },
    128: {
        "cls": {"stl10": 91.4, "cifar10": 94.1, "cifar100": 47.0,
                "imagenet": 18.8, "eurosat": 37.2},
        "ret": {"flickr_i2t": 69.2, "flickr_t2i": 54.8,
                "coco_i2t": 49.7, "coco_t2i": 37.2},
    },
    256: {
        "cls": {"stl10": 90.2, "cifar10": 93.0, "cifar100": 48.1,
                "imagenet": 20.2, "eurosat": 34.8},
        "ret": {"flickr_i2t": 72.2, "flickr_t2i": 58.0,
                "coco_i2t": 52.4, "coco_t2i": 39.2},
    },
    512: {
        "cls": {"stl10": 93.9, "cifar10": 96.9, "cifar100": 48.6,
                "imagenet": 21.2, "eurosat": 33.0},
        "ret": {"flickr_i2t": 75.5, "flickr_t2i": 60.2,
                "coco_i2t": 54.6, "coco_t2i": 41.2},
    },
    1024: {
        "cls": {"stl10": 93.7, "cifar10": 94.5, "cifar100": 48.5,
                "imagenet": 21.7, "eurosat": 33.1},
        "ret": {"flickr_i2t": 77.3, "flickr_t2i": 61.2,
                "coco_i2t": 55.3, "coco_t2i": 42.1},
    },
}

BASELINES = {
    "cls": {"stl10": 92.6, "cifar10": 95.0, "cifar100": 46.1,
            "imagenet": 22.1, "eurosat": 29.3},  # Linear+STR
    "ret": {"flickr_i2t": 62.9, "flickr_t2i": 48.9,
            "coco_i2t": 43.5, "coco_t2i": 32.8},  # FA
}

BASELINE_LABEL_CLS = "Linear$_{\\mathcal{R}_S}$"
BASELINE_LABEL_RET = "FreezeAlign"

# ── Compute averages ─────────────────────────────────────────────────
Ks = sorted(BA_RESULTS.keys())
cls_keys = list(BASELINES["cls"].keys())
ret_keys = list(BASELINES["ret"].keys())
avg_cls = [np.mean([BA_RESULTS[k]["cls"][d] for d in cls_keys]) for k in Ks]
avg_ret = [np.mean([BA_RESULTS[k]["ret"][d] for d in ret_keys]) for k in Ks]

baseline_cls = np.mean(list(BASELINES["cls"].values()))
baseline_ret = np.mean(list(BASELINES["ret"].values()))

# ── Palette (matching grouped bar chart) ─────────────────────────────
c_cls = "#264653"   # dark teal
c_ret = "#E76F51"   # burnt sienna

# ── Plot ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.8, 3.1))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# BA lines
ax.plot(range(len(Ks)), avg_cls, "o-", color=c_cls, linewidth=2,
        markersize=5, label="Avg. cls.", zorder=3)
ax.plot(range(len(Ks)), avg_ret, "s-", color=c_ret, linewidth=2,
        markersize=5, label="Avg. ret.", zorder=3)

# Baseline dashed lines
ax.axhline(baseline_cls, color=c_cls, linestyle="--", linewidth=1.2,
           alpha=0.5, label=f"Best baseline (cls.)")
ax.axhline(baseline_ret, color=c_ret, linestyle="--", linewidth=1.2,
           alpha=0.5, label=f"Best baseline (ret.)")

# Axes
ax.set_xticks(range(len(Ks)))
ax.set_xticklabels([str(k) for k in Ks], fontsize=12)
ax.tick_params(axis="y", labelsize=12)
ax.set_xlabel("Number of anchors (K)", fontsize=13)
ax.set_ylabel("Performance (%)", fontsize=13, labelpad=4)
ax.set_ylim(43, 61)

# Grid and spines
ax.grid(True, alpha=0.2, linewidth=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Trainable param count labels next to retrieval dots
param_labels = ["66K", "131K", "262K", "524K", "1M", "2.1M"]
param_offsets = [(16, 4), (8, -10), (8, -10), (8, -10), (0, -14), (0, -22)]
for i in range(len(Ks)):
    ax.annotate(param_labels[i], (i, avg_ret[i]),
                textcoords="offset points", xytext=param_offsets[i],
                ha="center", va="top", fontsize=9, color="black")

# Legend
ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 1.22),
          framealpha=1.0, edgecolor="#cccccc",
          handlelength=1.2, handletextpad=0.4, borderpad=0.3, labelspacing=0.3,
          ncol=2, columnspacing=1.0)

fig.tight_layout(pad=0.3)

# Save
out_dir = "drafts/figures"
fig.savefig(f"{out_dir}/k_sweep_ablation.png", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
fig.savefig(f"{out_dir}/k_sweep_ablation.pdf", bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print(f"Saved to {out_dir}/k_sweep_ablation.{{png,pdf}}")
