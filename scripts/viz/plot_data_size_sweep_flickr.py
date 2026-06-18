"""Data size sweep: BA vs FA on Flickr30k I2T and T2I.

Two-panel figure (I2T | T2I) for NeurIPS paper.

Usage:
    python scripts/viz/plot_data_size_sweep_flickr.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sizes = ["1K", "5K", "10K", "50K", "82K"]
x = np.arange(len(sizes))

flickr_ba_i2t = [16.9, 35.6, 44.6, 64.6, 69.6]
flickr_ba_t2i = [13.7, 26.7, 33.9, 50.1, 54.5]
flickr_fa_i2t = [13.5, 30.1, 37.6, 57.5, 61.3]
flickr_fa_t2i = [10.3, 23.7, 29.0, 44.5, 47.5]

c_ba = "#264653"
c_fa = "#E76F51"

fig, (ax_i, ax_t) = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
fig.patch.set_facecolor("white")

for ax, ba, fa, title in [(ax_i, flickr_ba_i2t, flickr_fa_i2t, "Flickr30k I2T"),
                            (ax_t, flickr_ba_t2i, flickr_fa_t2i, "Flickr30k T2I")]:
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

ax_i.set_ylabel("Retrieval R@1 (%)", fontsize=11)
ax_i.set_ylim(0, 78)
ax_t.set_ylim(0, 78)

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
fig.savefig(f"{out_dir}/retrieval_data_size_sweep_flickr.pdf",
            bbox_inches="tight", dpi=300, facecolor="white", edgecolor="none")
fig.savefig(f"{out_dir}/retrieval_data_size_sweep_flickr.png",
            bbox_inches="tight", dpi=300, facecolor="white", edgecolor="none")
plt.close()
print(f"Saved to {out_dir}/retrieval_data_size_sweep_flickr.{{pdf,png}}")
