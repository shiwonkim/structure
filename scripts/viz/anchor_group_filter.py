"""Filter COCO val samples by anchor-group membership and render a grid for each.

A sample passes if its top-K image-level anchors hit AT LEAST ONE anchor from
EACH provided group. The figure-anchors for that sample are the strongest hit
from each group, in group order — so the same colour slot stays bound to the
same group across all rendered samples.

Goal: show that distinct anchor groups play distinct semantic roles, and that
the same triplet of roles is reused across diverse images.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/viz/anchor_group_filter.py \
        --config configs/ba/vitl_roberta/token_k512.yaml \
        --ckpt 'serverB/.../checkpoint-epoch373.pth' \
        --groups '274,237|433,102,373|381,444,490' \
        --top-k 5 --n-samples 1500 \
        --out-dir drafts/figures/group_filtered
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from timm import create_model
from timm.data import resolve_data_config
from torchvision.models.feature_extraction import create_feature_extractor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.alignment import *  # noqa: F401,F403
from src.core.src.utils.loader import Loader, merge_dicts
from src.models.text.models import load_llm, load_tokenizer
from scripts.viz.anchor_cross_modal import (
    _ensure_rgb,
    get_image_anchor_attention,
    get_text_anchor_attention,
    plot_cross_modal_grid,
    select_top_anchors,
)
from scripts.viz.anchor_pair_finder import load_dataset_pool


def parse_groups(s):
    """'274,237|433,102,373|381,444,490' -> [[274,237],[433,102,373],[381,444,490]]."""
    groups = []
    for g in s.split("|"):
        ks = [int(x) for x in g.split(",") if x.strip()]
        if not ks:
            raise ValueError(f"empty group in {s!r}")
        groups.append(ks)
    return groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--groups", required=True,
                   help="Pipe-separated groups, e.g. '274,237|433,102,373|381,444,490'")
    p.add_argument("--top-k", type=int, default=5,
                   help="Sample's top-K img anchors are tested against the groups")
    p.add_argument("--min-groups", type=int, default=None,
                   help="Min # groups whose anchor must appear in top-K (default: ALL groups)")
    p.add_argument("--n-render-anchors", type=int, default=3,
                   help="How many of the hit anchors to render in the figure (top by attention)")
    p.add_argument("--strategy",
                   choices=["focused", "diverse", "consistent"],
                   default="focused",
                   help="select_top_anchors strategy used to define the per-sample top-K set")
    p.add_argument("--layer-img", type=int, default=23)
    p.add_argument("--layer-txt", type=int, default=24)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=1500)
    p.add_argument("--max-render", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", choices=["coco", "flickr30"], default="flickr30")
    p.add_argument("--out-dir", default="drafts/figures/group_filtered")
    p.add_argument("--output-json", default=None,
                   help="Defaults to <out-dir>/matches.json")
    args = p.parse_args()

    groups = parse_groups(args.groups)
    n_groups = len(groups)
    min_groups = args.min_groups if args.min_groups is not None else n_groups
    print(f"[groups] {groups}  min_groups={min_groups}/{n_groups}  top_k={args.top_k}")

    device = torch.device(f"cuda:{args.gpu}")
    with open(args.config) as f:
        cfg = Loader(f).get_single_data()
    cfg = merge_dicts(cfg.get("defaults", {}), cfg.get("overrides", {}))

    print(f"[ckpt] {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    alignment_image = ckpt["alignment_image"].eval().to(device)
    alignment_text = ckpt["alignment_text"].eval().to(device)

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

    print(f"[pool] sampling {args.n_samples} {args.dataset} images")
    pool = load_dataset_pool(args.dataset, args.n_samples, seed=args.seed)
    print(f"[pool] got {len(pool)} samples after caption filter")

    # Stage 1 — image + text forward, then filter using `select_top_anchors`
    # so the per-sample top-K matches anchor_cross_modal.py's notion of "top".
    all_a_img = []
    all_topN_focused = []  # ordered top-50 per sample for diagnostics
    DIAG_K = max(50, args.top_k)
    matches = []
    t0 = time.time()
    for s in pool:
        s["pil"] = Image.open(s["path"]).convert("RGB")
        img_attn, _ = get_image_anchor_attention(
            s["pil"], vision_model, image_transform, alignment_image,
            args.layer_img, device,
        )
        txt_attn, _, token_strs, _ = get_text_anchor_attention(
            s["caption"], tokenizer, language_model, alignment_text,
            args.layer_txt, device,
        )
        a_img = img_attn.max(axis=0)
        all_a_img.append(a_img)
        topN = select_top_anchors(img_attn, txt_attn,
                                  n=DIAG_K, strategy=args.strategy)
        all_topN_focused.append(topN)
        topk = set(topN[: args.top_k])
        hits_per_group = []
        for g in groups:
            inter = [k for k in g if k in topk]
            if inter:
                hits_per_group.append(max(inter, key=lambda k: a_img[k]))
        n_hits = len(hits_per_group)
        s["img_attn"] = img_attn
        s["txt_attn"] = txt_attn
        s["token_strs"] = token_strs
        s["topk"] = sorted(topk)
        s["focused_topN"] = topN
        s["a_img"] = a_img
        if n_hits < min_groups:
            continue
        chosen = sorted(hits_per_group, key=lambda k: -a_img[k])[: args.n_render_anchors]
        s["anchors"] = chosen
        s["all_hits"] = hits_per_group
        s["n_hits"] = n_hits
        s["group_score"] = float(sum(a_img[k] for k in chosen))
        matches.append(s)
    print(f"[filter] {len(matches)} / {len(pool)} samples matched "
          f"(stage 1: {time.time() - t0:.1f}s, strategy={args.strategy})")

    # ── Diagnostics ─────────────────────────────────────────────
    A = np.stack(all_a_img)  # (N, K)
    N, K = A.shape
    mean_a = A.mean(axis=0)
    print(f"\n[diag] pool size={N}, K={K}")
    print(f"[diag] mean(max-patch attention) over pool: "
          f"min={mean_a.min():.4f}  median={np.median(mean_a):.4f}  "
          f"max={mean_a.max():.4f}")
    from collections import Counter
    print(f"[diag] thresholds use {args.strategy} top-K (select_top_anchors)")
    for K_thresh in (5, 10, 20, 50):
        if K_thresh > DIAG_K:
            continue
        topk_sets = [set(t[:K_thresh]) for t in all_topN_focused]
        anchor_counts = np.zeros(K, dtype=int)
        for ts in topk_sets:
            for k in ts:
                anchor_counts[k] += 1
        ngroup_counter = Counter()
        for ts in topk_sets:
            n = sum(1 for g in groups if set(g) & ts)
            ngroup_counter[n] += 1
        group_hit_counts = []
        for g in groups:
            n_hit = sum(1 for ts in topk_sets if set(g) & ts)
            group_hit_counts.append(n_hit)
        breakdown = "  ".join(
            f"{n}/{n_groups}={ngroup_counter.get(n, 0)}"
            for n in range(n_groups, -1, -1)
        )
        print(f"[diag] K={K_thresh:>3}: per-group hits={group_hit_counts}  "
              f"groups-hit-dist [{breakdown}]")
        for gi, g in enumerate(groups):
            details = ", ".join(
                f"#{k} (top{K_thresh}={anchor_counts[k]}/{N}, "
                f"meanAtt={mean_a[k]:.4f})"
                for k in g
            )
            print(f"        group{gi + 1} → {details}")

    matches.sort(key=lambda s: (-s["n_hits"], -s["group_score"]))
    matches = matches[: args.max_render]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 2 — grid render for matches only (text attn already cached)
    meta_out = []
    t1 = time.time()
    for r, s in enumerate(matches):
        sample_for_grid = [{
            "image": s["pil"], "caption": s["caption"],
            "img_attn": s["img_attn"], "txt_attn": s["txt_attn"],
            "token_strs": s["token_strs"], "anchor_indices": s["anchors"],
        }]
        anchor_str = "-".join(str(a) for a in s["anchors"])
        out_path = out_dir / (
            f"sample_{r + 1:02d}_h{s['n_hits']}_score{s['group_score']:.3f}"
            f"_idx{s['idx']}_a{anchor_str}.png"
        )
        plot_cross_modal_grid(sample_for_grid, str(out_path))
        meta_out.append({
            "rank": r + 1,
            "n_hits": s["n_hits"],
            "n_groups": n_groups,
            "group_score": s["group_score"],
            "image_idx": s["idx"],
            "image_id": s["image_id"],
            "path": s["path"],
            "caption": s["caption"],
            "render_anchors": s["anchors"],
            "all_hit_anchors": s["all_hits"],   # one per hit group
            "topk_anchors": s["topk"],
        })
    print(f"[render] {len(matches)} grids in {time.time() - t1:.1f}s")

    json_path = args.output_json or str(out_dir / "matches.json")
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"[done] metadata -> {json_path}")


if __name__ == "__main__":
    main()
