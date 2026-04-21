#!/usr/bin/env python3
"""Stacked bar chart: CLS → Token performance gain.

Broken y-axis, grid behind bars, uniform colors.
Easy to add new methods — just append to METHODS dict.

Usage:
    python drafts/figures/plot_stacked_bar.py
    # outputs: drafts/figures/stacked_bar_cls_vs_token.{pdf,png}
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ============================================================
# DATA — add new methods here
# Format: "Label": {"DATASET": (CLS_value, Token_value), ...}
# ============================================================
METHODS = {
    "Linear":     {"Flickr I2T": (60.4, 63.8), "Flickr T2I": (47.0, 50.3)},
    "Linear+STR": {"Flickr I2T": (62.8, 66.2), "Flickr T2I": (48.0, 49.6)},
    "MLP":        {"Flickr I2T": (60.4, 64.2), "Flickr T2I": (46.2, 51.6)},
    "MLP+STR":    {"Flickr I2T": (61.3, 64.3), "Flickr T2I": (47.6, 49.7)},
    # --- add more baselines below ---
    # "FreezeAlign": {"Flickr I2T": (XX, YY), "Flickr T2I": (XX, YY)},
    "BA K=512":   {"Flickr I2T": (59.3, 75.5), "Flickr T2I": (46.3, 60.2)},
}

# Dataset configs: (title, y_break_top, y_break_bot, y_top)
DATASET_CFG = {
    "Flickr I2T": ("Flickr30k I2T R@1", (0, 3), (52, 80)),
    "Flickr T2I": ("Flickr30k T2I R@1", (0, 3), (40, 66)),
}

# ============================================================
# STYLE
# ============================================================
COLOR_BASE = "#d5d8dc"
COLOR_GAIN = "#2980b9"
EDGE_COLOR = "#5d6d7e"
BAR_WIDTH = 0.55
FIG_WIDTH = 10
FIG_HEIGHT = 4.2


def plot():
    method_names = list(METHODS.keys())
    dataset_keys = list(DATASET_CFG.keys())
    n_methods = len(method_names)
    x = np.arange(n_methods)

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    fig.patch.set_facecolor('white')

    for col, ds_key in enumerate(dataset_keys):
        title, bot_lim, top_lim = DATASET_CFG[ds_key]

        cls_vals = [METHODS[m][ds_key][0] for m in method_names]
        tok_vals = [METHODS[m][ds_key][1] for m in method_names]
        deltas   = [t - c for c, t in zip(cls_vals, tok_vals)]

        # Two axes: top (main) and bottom (break stub)
        ax_top = fig.add_axes([0.08 + col * 0.48, 0.15, 0.40, 0.72])
        ax_bot = fig.add_axes([0.08 + col * 0.48, 0.08, 0.40, 0.04])

        for ax in [ax_top, ax_bot]:
            ax.grid(axis='y', alpha=0.2, linestyle='--', zorder=0)
            ax.bar(x, cls_vals, BAR_WIDTH,
                   color=COLOR_BASE, edgecolor=EDGE_COLOR, linewidth=0.8, zorder=2)
            ax.bar(x, deltas, BAR_WIDTH, bottom=cls_vals,
                   color=COLOR_GAIN, edgecolor=EDGE_COLOR, linewidth=0.8, zorder=2)
            ax.set_xticks(x)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_facecolor('white')
            ax.patch.set_alpha(1.0)

        ax_top.set_ylim(*top_lim)
        ax_bot.set_ylim(*bot_lim)

        ax_top.spines['bottom'].set_visible(False)
        ax_bot.spines['top'].set_visible(False)
        ax_top.tick_params(bottom=False, labelbottom=False)
        ax_bot.set_xticklabels(method_names, fontsize=10)
        ax_bot.set_yticks([0])

        # Break marks — scale y-component by axis height ratio so both
        # sets have the same visual angle.
        d = 0.010
        h_top, h_bot = 0.72, 0.04  # must match add_axes heights above
        d_y_bot = d * (h_top / h_bot)  # scale bottom y to match top angle
        kwargs = dict(color='k', clip_on=False, linewidth=1)
        ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kwargs)
        ax_bot.plot((-d, +d), (1 - d_y_bot, 1 + d_y_bot), transform=ax_bot.transAxes, **kwargs)
        ax_bot.plot((1 - d, 1 + d), (1 - d_y_bot, 1 + d_y_bot), transform=ax_bot.transAxes, **kwargs)

        # Annotations
        for i in range(n_methods):
            if cls_vals[i] >= top_lim[0]:
                ax_top.text(x[i], cls_vals[i] - 1.0, f"{cls_vals[i]:.1f}",
                            ha='center', va='top', fontsize=8.5, color='#5d6d7e', zorder=3)
            ax_top.text(x[i], cls_vals[i] + deltas[i] / 2, f"+{deltas[i]:.1f}",
                        ha='center', va='center', fontsize=9, color='white', zorder=3)
            ax_top.text(x[i], tok_vals[i] + 0.5, f"{tok_vals[i]:.1f}",
                        ha='center', va='bottom', fontsize=10, color='#2c3e50', zorder=3)

        ax_top.set_title(title, fontsize=13, pad=8)
        ax_top.set_ylabel("R@1 (%)", fontsize=11)

    # Legend on first panel
    legend_elements = [
        Patch(facecolor=COLOR_BASE, edgecolor=EDGE_COLOR, label='CLS (base)'),
        Patch(facecolor=COLOR_GAIN, edgecolor=EDGE_COLOR, label='Token gain'),
    ]
    fig.axes[0].legend(handles=legend_elements, loc='upper left', fontsize=9,
                       framealpha=1.0, edgecolor='#cccccc')

    out_base = "drafts/figures/stacked_bar_cls_vs_token"
    plt.savefig(f"{out_base}.pdf", bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    plt.savefig(f"{out_base}.png", bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    print(f"Saved to {out_base}.{{pdf,png}}")

    # Print table
    print(f"\n{'Method':15s}", end="")
    for ds in dataset_keys:
        print(f" | {ds:>12s} CLS  Delta Token", end="")
    print()
    print("-" * (15 + len(dataset_keys) * 35))
    for m in method_names:
        print(f"{m:15s}", end="")
        for ds in dataset_keys:
            c, t = METHODS[m][ds]
            print(f" | {c:12.1f} {t-c:+6.1f} {t:5.1f}", end="")
        print()


if __name__ == "__main__":
    plot()
