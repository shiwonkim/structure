"""Data size sweep: BA vs FA retrieval R@1 across training sizes.

Two-panel figure (Flickr30k | COCO) for NeurIPS paper.

Usage:
    python scripts/viz/plot_data_size_sweep.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────
sizes = ["1K", "5K", "10K", "50K", "82K"]
x = np.arange(len(sizes))

flickr = {
    "ba_i2t":  [16.9, 35.6, 44.6, 64.6, 69.6],
    "ba_t2i":  [13.7, 26.7, 33.9, 50.1, 54.5],
    "fa_i2t":  [13.5, 30.1, 37.6, 57.5, 61.3],
    "fa_t2i":  [10.3, 23.7, 29.0, 44.5, 47.5],
}

coco = {
    "ba_i2t":  [10.4, 24.0, 30.7, 47.2, 51.5],
    "ba_t2i":  [8.5, 18.2, 23.2, 34.7, 38.6],
    "fa_i2t":  [8.5, 19.6, 24.1, 40.6, 44.2],
    "fa_t2i":  [6.9, 16.4, 20.0, 30.2, 32.5],
}

# ── Style ─────────────────────────────────────────────────────────────
c_ba = "#264653"   # dark teal
c_fa = "#E76F51"   # burnt sienna

fig, (ax_f, ax_c) = plt.subplots(1, 2, figsize=(6.0, 3.4), sharey=True)
fig.patch.set_facecolor("white")

for ax, data, title in [(ax_f, flickr, "Flickr30k"), (ax_c, coco, "COCO Karpathy")]:
    ax.set_facecolor("white")

    # BA: solid for I2T, dashed for T2I
    ax.plot(x, data["ba_i2t"], "o-", color=c_ba, linewidth=1.8,
            markersize=4.5, label="BA (ours)", zorder=3)
    ax.plot(x, data["ba_t2i"], "o--", color=c_ba, linewidth=1.8,
            markersize=4.5, zorder=3)

    # FA: solid for I2T, dashed for T2I
    ax.plot(x, data["fa_i2t"], "s-", color=c_fa, linewidth=1.8,
            markersize=4.5, label="Best baseline (ret.)", zorder=3)
    ax.plot(x, data["fa_t2i"], "s--", color=c_fa, linewidth=1.8,
            markersize=4.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=10)
    ax.set_xlabel("Training data size", fontsize=11)
    ax.set_title(title, fontsize=11, pad=6)
    ax.tick_params(axis="y", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)

ax_f.set_ylabel("Retrieval R@1 (%)", fontsize=11)

# Shared y-limits
ax_f.set_ylim(0, 78)
ax_c.set_ylim(0, 78)

# Legend: methods only, with line style explanation
# Add invisible dummy lines for I2T/T2I distinction
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=c_ba, linewidth=1.8, marker="o", markersize=4.5,
           label="BA (ours)"),
    Line2D([0], [0], color=c_fa, linewidth=1.8, marker="s", markersize=4.5,
           label="Best baseline (ret.)"),
    Line2D([0], [0], color="gray", linewidth=1.2, linestyle="-",
           label="I2T"),
    Line2D([0], [0], color="gray", linewidth=1.2, linestyle="--",
           label="T2I"),
]

fig.legend(handles=legend_elements, loc="upper center",
           bbox_to_anchor=(0.5, 1.12), ncol=4, fontsize=9,
           framealpha=1.0, edgecolor="#cccccc",
           handlelength=1.8, handletextpad=0.5, columnspacing=1.5)

fig.tight_layout(pad=0.4)
fig.subplots_adjust(top=0.85)

out_dir = "drafts/figures"
fig.savefig(f"{out_dir}/retrieval_data_size_sweep_combined.pdf",
            bbox_inches="tight", dpi=300, facecolor="white", edgecolor="none")
fig.savefig(f"{out_dir}/retrieval_data_size_sweep_combined.png",
            bbox_inches="tight", dpi=300, facecolor="white", edgecolor="none")
plt.close()

print("Design choices:")
print("  - Two-panel layout: Flickr30k (left) | COCO (right)")
print("  - Shared y-axis (0-78%) for direct comparison")
print("  - Methods: color (teal=BA, sienna=FA)")
print("  - Directions: line style (solid=I2T, dashed=T2I)")
print("  - Compact legend above both panels, 4 entries in one row")
print("  - Green delta annotations at 82K showing BA gain over FA")
print("  - Categorical x-axis: 1K, 5K, 10K, 50K, 82K")
print("  - Light grid, clean spines, white background")
print(f"\nSaved to {out_dir}/retrieval_data_size_sweep_combined.{{pdf,png}}")
