"""Data size sweep: BA vs FA average retrieval R@1 across training sizes.

Two-panel figure (Flickr30k | COCO), averaging I2T and T2I per method.

Usage:
    python scripts/viz/plot_data_size_sweep_avg.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sizes = ["1K", "5K", "10K", "50K", "82K"]
x = np.arange(len(sizes))

# Average of I2T and T2I
flickr_ba = [(16.9+13.7)/2, (35.6+26.7)/2, (44.6+33.9)/2, (64.6+50.1)/2, (69.6+54.5)/2]
flickr_fa = [(13.5+10.3)/2, (30.1+23.7)/2, (37.6+29.0)/2, (57.5+44.5)/2, (61.3+47.5)/2]
coco_ba   = [(10.4+8.5)/2, (24.0+18.2)/2, (30.7+23.2)/2, (47.2+34.7)/2, (51.5+38.6)/2]
coco_fa   = [(8.5+6.9)/2, (19.6+16.4)/2, (24.1+20.0)/2, (40.6+30.2)/2, (44.2+32.5)/2]

c_ba = "#264653"
c_fa = "#E76F51"

fig, (ax_f, ax_c) = plt.subplots(1, 2, figsize=(6.0, 3.4), sharey=True)
fig.patch.set_facecolor("white")

for ax, ba, fa, title in [(ax_f, flickr_ba, flickr_fa, "Flickr30k"),
                            (ax_c, coco_ba, coco_fa, "COCO Karpathy")]:
    ax.set_facecolor("white")
    ax.plot(x, ba, "o-", color=c_ba, linewidth=1.8, markersize=4.5,
            label="BA (ours)", zorder=3)
    ax.plot(x, fa, "s-", color=c_fa, linewidth=1.8, markersize=4.5,
            label="Best baseline (ret.)", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=10)
    ax.set_xlabel("Training data size", fontsize=11)
    ax.set_title(title, fontsize=11, pad=6)
    ax.tick_params(axis="y", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)

ax_f.set_ylabel("Avg. Retrieval R@1 (%)", fontsize=11)
ax_f.set_ylim(0, 70)
ax_c.set_ylim(0, 70)

fig.legend(handles=[
    Line2D([0], [0], color=c_ba, linewidth=1.8, marker="o", markersize=4.5,
           label="BA (ours)"),
    Line2D([0], [0], color=c_fa, linewidth=1.8, marker="s", markersize=4.5,
           label="Best baseline (ret.)"),
], loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, fontsize=9.5,
   framealpha=1.0, edgecolor="#cccccc",
   handlelength=1.8, handletextpad=0.5, columnspacing=1.5)

fig.tight_layout(pad=0.4)
fig.subplots_adjust(top=0.85)

out_dir = "drafts/figures"
fig.savefig(f"{out_dir}/retrieval_data_size_sweep_avg.pdf",
            bbox_inches="tight", dpi=300, facecolor="white", edgecolor="none")
fig.savefig(f"{out_dir}/retrieval_data_size_sweep_avg.png",
            bbox_inches="tight", dpi=300, facecolor="white", edgecolor="none")
plt.close()
print(f"Saved to {out_dir}/retrieval_data_size_sweep_avg.{{pdf,png}}")

# Print averages
print("\nAvg retrieval R@1:")
print(f"{'Size':>6s}  {'Flickr BA':>10s}  {'Flickr FA':>10s}  {'COCO BA':>10s}  {'COCO FA':>10s}")
for i, s in enumerate(sizes):
    print(f"{s:>6s}  {flickr_ba[i]:>10.1f}  {flickr_fa[i]:>10.1f}  {coco_ba[i]:>10.1f}  {coco_fa[i]:>10.1f}")
