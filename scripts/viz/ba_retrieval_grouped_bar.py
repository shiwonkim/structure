"""Grouped bar chart for narrow NeurIPS wrapfigure/minipage layout.
CLS BA K=512 vs Token BA K=512 (Mean Pool) vs Token BA K=512 (CAP)
on Flickr I2T, Flickr T2I, COCO I2T, COCO T2I retrieval R@1.

Designed for ~0.45\textwidth column insertion. All fonts, bar widths,
and spacing are tuned for readability after downscaling."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# -- Data --
groups = ["Flickr\nI2T", "Flickr\nT2I", "COCO\nI2T", "COCO\nT2I"]
cls_ba = [59.3, 46.3, 40.1, 29.5]
token_mean = [63.5, 51.2, 44.9, 33.6]
token_cap = [75.5, 60.2, 54.6, 41.2]

x = np.arange(len(groups))
width = 0.28  # thicker bars for narrow layout

# -- Palette --
c1 = "#264653"   # dark teal
c2 = "#E76F51"   # burnt sienna
c3 = "#2A9D8F"   # seafoam green

# -- Figure: narrow and tall for wrapfigure --
fig, ax = plt.subplots(figsize=(3.8, 3.1))
fig.patch.set_facecolor('white')

ax.bar(x - width, cls_ba, width, label="CLS only", color=c1, edgecolor="white", linewidth=0.4)
ax.bar(x, token_mean, width, label=" + Patch/token seq.", color=c3, edgecolor="white", linewidth=0.4, hatch="//", zorder=2)
ax.bar(x + width, token_cap, width, label=" + Cross-attn pool", color=c2, edgecolor="white", linewidth=0.4)

# -- Axes --
ax.set_ylabel("R@1 (%)", fontsize=13, labelpad=4)
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=12, linespacing=0.85)
ax.tick_params(axis='y', labelsize=12)
ax.set_ylim(0, 82)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_facecolor('white')

# -- Legend: compact, inside plot --
ax.legend(fontsize=10, loc="upper right", bbox_to_anchor=(1.0, 1.25),
          framealpha=1.0, edgecolor='#cccccc',
          handlelength=1.2, handletextpad=0.4, borderpad=0.3, labelspacing=0.3)

# -- Delta labels (CAP gain over mean pool) --
for i in range(len(groups)):
    delta = token_cap[i] - token_mean[i]
    ax.text(x[i] + width, token_cap[i] + 0.8,
            f"+{delta:.1f}", ha="center", va="bottom", fontsize=11,
            color="#2e7d32", fontweight="bold")

fig.tight_layout(pad=0.3)
fig.savefig("drafts/figures/ba_retrieval_grouped_bar.pdf", bbox_inches="tight", dpi=300,
            facecolor="white", edgecolor="none")
fig.savefig("drafts/figures/ba_retrieval_grouped_bar.png", bbox_inches="tight", dpi=300,
            facecolor="white", edgecolor="none")
print("Saved to drafts/figures/ba_retrieval_grouped_bar.{pdf,png}")
