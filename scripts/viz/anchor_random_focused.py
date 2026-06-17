"""Render grids for N random COCO val samples using the script's own
anchor-selection heuristic (focused / diverse / consistent).

Each sample's 3 anchors are picked by select_top_anchors() in
anchor_cross_modal.py — independent per sample, no group constraint, no
cross-sample anchor sharing.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/viz/anchor_random_focused.py \
        --config configs/ba/vitl_roberta/token_k512.yaml \
        --ckpt 'serverB/.../checkpoint-epoch373.pth' \
        --layer-img 23 --layer-txt 24 --gpu 0 \
        --n-samples 10 --n-anchors 3 --strategy focused \
        --out-dir drafts/figures/random_focused
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--layer-img", type=int, default=23)
    p.add_argument("--layer-txt", type=int, default=24)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--n-anchors", type=int, default=3)
    p.add_argument("--strategy",
                   choices=["focused", "diverse", "consistent"],
                   default="focused")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", choices=["coco", "flickr30"], default="flickr30")
    p.add_argument("--out-dir", default="drafts/figures/random_focused")
    args = p.parse_args()

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

    pool = load_dataset_pool(args.dataset, args.n_samples, seed=args.seed)
    print(f"[pool] {len(pool)} {args.dataset} samples (strategy={args.strategy})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_out = []
    t0 = time.time()
    for r, s in enumerate(pool):
        pil = Image.open(s["path"]).convert("RGB")
        img_attn, _ = get_image_anchor_attention(
            pil, vision_model, image_transform, alignment_image,
            args.layer_img, device,
        )
        txt_attn, _, token_strs, _ = get_text_anchor_attention(
            s["caption"], tokenizer, language_model, alignment_text,
            args.layer_txt, device,
        )
        anchors = select_top_anchors(
            img_attn, txt_attn, n=args.n_anchors, strategy=args.strategy,
        )
        sample_for_grid = [{
            "image": pil, "caption": s["caption"],
            "img_attn": img_attn, "txt_attn": txt_attn,
            "token_strs": token_strs, "anchor_indices": anchors,
        }]
        anchor_str = "-".join(str(a) for a in anchors)
        out_path = out_dir / (
            f"sample_{r + 1:02d}_idx{s['idx']}_a{anchor_str}.png"
        )
        plot_cross_modal_grid(sample_for_grid, str(out_path))
        meta_out.append({
            "rank": r + 1,
            "image_idx": s["idx"],
            "image_id": s["image_id"],
            "path": s["path"],
            "caption": s["caption"],
            "anchors": anchors,
        })
    print(f"[done] {len(pool)} grids in {time.time() - t0:.1f}s")
    with open(out_dir / "matches.json", "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"[meta] {out_dir / 'matches.json'}")


if __name__ == "__main__":
    main()
