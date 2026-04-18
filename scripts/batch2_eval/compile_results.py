"""Compile batch 2 eval logs into results/batch2_full_results.md.

Reads:
  - logs/batch2_eval/{LABEL}_{native,clsfallback}.log   (zero-shot + retrieval)
  - logs/batch2_eval/seg_{LABEL}_{dataset}.log          (segmentation)

Parses:
  - Classification  : "Dataset - top1_acc_micro: X, ..."
  - Retrieval        : "coco/I2T-R@1: 0.xx" / wandb summary
  - Segmentation    : table rows "method  strategy  miou-fg%  miou-all%"

Outputs:
  results/batch2_full_results.md  — 3 markdown tables
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


# Parallel lists because order matters for table display.
# CSA static results (no checkpoint, from existing CSVs).
CSA_STATIC = {
    "CSA d=256": {
        "cifar10": 0.7232, "cifar100": 0.2495, "stl10": 0.8278, "mnist": 0.1061,
    },
    "CSA d=256 + STR": {
        "cifar10": 0.8066, "cifar100": 0.3185, "stl10": 0.8999, "mnist": 0.1222,
    },
}
CSA_RETRIEVAL = {
    "CSA d=256":     {("coco", "I2T", "R@1"): 0.0398, ("coco", "T2I", "R@1"): 0.0257},
    "CSA d=256 + STR": {("coco", "I2T", "R@1"): 0.0416, ("coco", "T2I", "R@1"): 0.0297},
}

METHODS = [
    # label, display, mode (native/clsfallback), ckpt type
    ("01_linear",           "Linear d=512",                "native",      "cls"),
    ("02_linear_struct",    "Linear d=512 + STR",          "native",      "cls"),
    ("03_mlp",              "MLP d=512",                   "native",      "cls"),
    ("04_mlp_struct",       "MLP d=512 + STR",             "native",      "cls"),
    ("_csa_d256",           "CSA d=256",                   "static",      "cls"),
    ("_csa_d256_struct",    "CSA d=256 + STR",             "static",      "cls"),
    ("05_cls_ba_k128",      "CLS BA K=128",                "native",      "cls"),
    ("06_cls_ba_k256",      "CLS BA K=256",                "native",      "cls"),
    ("07_cls_ba_k512",      "CLS BA K=512",                "native",      "cls"),
    ("08_token_ba_k128",    "Token BA K=128 (token)",      "native",      "token"),
    ("08_token_ba_k128",    "Token BA K=128 (CLS-fb)",     "clsfallback", "token"),
    ("09_token_ba_k256",    "Token BA K=256 (token)",      "native",      "token"),
    ("09_token_ba_k256",    "Token BA K=256 (CLS-fb)",     "clsfallback", "token"),
    ("10_token_ba_k512",    "Token BA K=512 (token)",      "native",      "token"),
    ("10_token_ba_k512",    "Token BA K=512 (CLS-fb)",     "clsfallback", "token"),
    ("11_fa_d512",          "FreezeAlign (token)",         "native",      "token"),
    ("11_fa_d512",          "FreezeAlign (CLS-fb)",        "clsfallback", "token"),
    ("12_fa_struct",        "FreezeAlign+STR (token)",     "native",      "token"),
    ("12_fa_struct",        "FreezeAlign+STR (CLS-fb)",    "clsfallback", "token"),
    ("13_token_ba_k128_p",  "Token BA K=128 + Prior (token)",   "native",      "token"),
    ("13_token_ba_k128_p",  "Token BA K=128 + Prior (CLS-fb)",  "clsfallback", "token"),
    ("14_token_ba_k256_p",  "Token BA K=256 + Prior (token)",   "native",      "token"),
    ("14_token_ba_k256_p",  "Token BA K=256 + Prior (CLS-fb)",  "clsfallback", "token"),
    ("15_token_ba_k512_p",  "Token BA K=512 + Prior (token)",   "native",      "token"),
    ("15_token_ba_k512_p",  "Token BA K=512 + Prior (CLS-fb)",  "clsfallback", "token"),
]

SEG_RUNS = [
    ("seg_token_ba_k128",   "Token BA K=128"),
    ("seg_token_ba_k256",   "Token BA K=256"),
    ("seg_token_ba_k512",   "Token BA K=512"),
    ("seg_fa_d512",         "FreezeAlign"),
    ("seg_fa_struct",       "FreezeAlign+STR"),
    ("seg_token_ba_k128_p", "Token BA K=128 + Prior"),
    ("seg_token_ba_k256_p", "Token BA K=256 + Prior"),
    ("seg_token_ba_k512_p", "Token BA K=512 + Prior"),
]

ZS_DATASETS = [
    # display, parser key (case-insensitive)
    ("CIFAR-10",   "cifar10"),
    ("CIFAR-100",  "cifar100"),
    ("STL-10",     "stl10"),
    ("MNIST",      "mnist"),
    ("DTD",        "dtd"),
    ("Flowers",    "flowers"),
    ("GTSRB",      "gtsrb"),
    ("Country211", "country211"),
]

RT_DATASETS = [
    ("COCO I2T R@1",    ("coco", "I2T", "R@1")),
    ("COCO T2I R@1",    ("coco", "T2I", "R@1")),
    ("Flickr I2T R@1",  ("flickr30", "I2T", "R@1")),
    ("Flickr T2I R@1",  ("flickr30", "T2I", "R@1")),
]

SEG_DATASETS = [
    ("VOC",        "voc2012"),
    ("Context",    "pascal_context"),
    ("ADE20K",     "ade20k"),
    ("Cityscapes", "cityscapes"),
]


def parse_classification_log(path: Path) -> dict[str, float]:
    """Return {'cifar10': top1_micro, 'dtd': ...} parsed from rerun_eval stdout."""
    out: dict[str, float] = {}
    if not path.exists():
        return out
    rx = re.compile(
        r"([A-Za-z0-9_]+)\s+-\s+top1_acc_micro:\s*([0-9.]+)"
    )
    with open(path, errors="replace") as f:
        for line in f:
            m = rx.search(line)
            if m:
                name = m.group(1).lower()
                val = float(m.group(2))
                out[name] = val
    return out


def parse_retrieval_log(path: Path) -> dict[tuple[str, str, str], float]:
    """Return {(dataset, i2t/t2i, R@k): value} parsed from rerun_eval retrieval output.

    Trainer emits one line per dataset with all R@k values:
        "Coco - I2T-R@1: 0.085, I2T-R@5: 0.230, ..., T2I-R@1: 0.062, ..."
        "Flickr30 - I2T-R@1: ..."
    """
    out: dict[tuple[str, str, str], float] = {}
    if not path.exists():
        return out
    line_rx = re.compile(r"(Coco|Flickr30)\s+-\s+(.*)$")
    metric_rx = re.compile(r"(I2T|T2I)-R@(\d+):\s*([0-9.]+)")
    with open(path, errors="replace") as f:
        for line in f:
            ml = line_rx.search(line)
            if not ml:
                continue
            dataset = ml.group(1).lower()  # coco, flickr30
            for mm in metric_rx.finditer(ml.group(2)):
                key = (dataset, mm.group(1), f"R@{mm.group(2)}")
                out[key] = float(mm.group(3))
    return out


def parse_seg_log(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Return {(method, strategy): {'miou_fg': X, 'miou_all': Y}} from seg log."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    if not path.exists():
        return out
    # Table rows look like:
    #   direct_cosine        raw            0.88%      2.48%
    # (4 whitespace-separated tokens after the method name)
    rx = re.compile(
        r"^\s*(direct_cosine|freezealign|anchor_codebook|attention_map)\s+"
        r"(raw|ensemble)\s+"
        r"([0-9.]+)%\s+([0-9.]+)%"
    )
    with open(path, errors="replace") as f:
        for line in f:
            m = rx.match(line)
            if m:
                out[(m.group(1), m.group(2))] = {
                    "miou_fg": float(m.group(3)),
                    "miou_all": float(m.group(4)),
                }
    return out


def fmt_pct(v: float | None, digits: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.{digits}f}"


def fmt_num(v: float | None, digits: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def build_classification_table(log_dir: Path) -> str:
    header = ["Method", "Mode"] + [d[0] for d in ZS_DATASETS]
    rows: list[list[str]] = []
    for label, display, mode, _kind in METHODS:
        if mode == "static":
            metrics = CSA_STATIC.get(display, {})
        else:
            log_path = log_dir / f"{label}_{mode}.log"
            metrics = parse_classification_log(log_path)
        if not metrics:
            rows.append([display, "", *["—"] * len(ZS_DATASETS)])
            continue
        mode_disp = "token" if mode == "native" and _kind == "token" else (
            "CLS" if mode == "clsfallback" else "—"
        )
        row = [display, mode_disp]
        for _disp, key in ZS_DATASETS:
            row.append(fmt_pct(metrics.get(key)))
        rows.append(row)
    return make_md_table(header, rows)


def build_retrieval_table(log_dir: Path) -> str:
    # Only show one row per checkpoint (retrieval doesn't change with
    # token_level_zero_shot). Skip CLS-fallback rows for token methods.
    header = ["Method"] + [d[0] for d in RT_DATASETS]
    rows = []
    seen_labels: set[str] = set()
    for label, display, mode, _kind in METHODS:
        if mode == "clsfallback":
            continue
        if label in seen_labels:
            continue
        seen_labels.add(label)
        # Strip mode suffix from display for retrieval table
        clean_display = display.replace(" (token)", "").replace(" (CLS-fb)", "")
        if mode == "static":
            metrics = CSA_RETRIEVAL.get(clean_display, {})
        else:
            log_path = log_dir / f"{label}_{mode}.log"
            metrics = parse_retrieval_log(log_path)
        if not metrics:
            rows.append([clean_display, *["—"] * len(RT_DATASETS)])
            continue
        row = [clean_display]
        for _d, key in RT_DATASETS:
            row.append(fmt_pct(metrics.get(key)))
        rows.append(row)
    return make_md_table(header, rows)


def build_segmentation_table(log_dir: Path) -> str:
    # For each (seg_label, strategy, method) across 4 datasets → one row.
    header = ["Checkpoint", "Method", "Strategy"] + [d[0] for d in SEG_DATASETS]
    rows: list[list[str]] = []
    for label, display in SEG_RUNS:
        # Collect parsed results per dataset
        ds_results: dict[str, dict[tuple[str, str], dict[str, float]]] = {}
        for ds_disp, ds_key in SEG_DATASETS:
            ds_results[ds_key] = parse_seg_log(log_dir / f"{label}_{ds_key}.log")
        # Build a row per (method, strategy) combo actually present
        methods_seen: set[tuple[str, str]] = set()
        for d in ds_results.values():
            methods_seen.update(d.keys())
        for method_name, strategy in sorted(methods_seen):
            row = [display, method_name, strategy]
            for _ds_disp, ds_key in SEG_DATASETS:
                m = ds_results[ds_key].get((method_name, strategy))
                row.append(fmt_num(m["miou_fg"]) if m else "—")
            rows.append(row)
    return make_md_table(header, rows)


def make_md_table(header: list[str], rows: list[list[str]]) -> str:
    col_widths = [len(h) for h in header]
    for r in rows:
        for i, v in enumerate(r):
            col_widths[i] = max(col_widths[i], len(str(v)))
    def fmt_row(cells):
        return "| " + " | ".join(
            str(c).ljust(col_widths[i]) for i, c in enumerate(cells)
        ) + " |"
    lines = [fmt_row(header), "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"]
    for r in rows:
        lines.append(fmt_row(r))
    return "\n".join(lines)


def main():
    root = Path(__file__).resolve().parent.parent.parent
    log_dir = root / "logs" / "batch2_eval"
    out_path = root / "results" / "batch2_full_results.md"

    cls_table = build_classification_table(log_dir)
    rt_table = build_retrieval_table(log_dir)
    seg_table = build_segmentation_table(log_dir)

    md = [
        "# Batch 2 Full Evaluation — ViT-S + MiniLM, COCO 2014",
        "",
        "All checkpoints trained on COCO 2014 train (82,783 images). Layer pair (img=11, txt=6).",
        "",
        "## Table 1 — Zero-Shot Classification (Top-1, %)",
        "",
        "Mode: `token` = token-level CAP/mean-pool eval path; `CLS` = 2D CLS-fallback branch. "
        "CLS methods have only one mode (dash). CSA runs excluded — no persistent checkpoint.",
        "",
        cls_table,
        "",
        "## Table 2 — Retrieval (R@1, %)",
        "",
        "COCO 2014 val and Flickr30k test (Karpathy split).",
        "",
        rt_table,
        "",
        "## Table 3 — Zero-Shot Segmentation (mIoU-fg, %)",
        "",
        "Token BA + FreezeAlign only. `direct_cosine` is the raw-encoder baseline "
        "(no alignment). `pascal_context` is computed on a 2000-image subset of trainval "
        "(no official val.txt); other datasets use the full val split.",
        "",
        "Reference: FreezeAlign paper reports **VOC 31.37** / **Context 24.61** (different encoder — CLIP-L).",
        "",
        seg_table,
    ]
    out_path.write_text("\n".join(md) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
