"""Batch-render grid figures for every pair listed in a finder JSON.

Reads the metadata JSON produced by anchor_pair_finder.py and produces one
PNG (and PDF) per pair using anchor_cross_modal.plot_cross_modal_grid.

Forwards each unique image at most once (cache keyed by (path, caption)).

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/viz/anchor_pair_grid_batch.py \
        --config configs/ba/vitl_roberta/token_k512.yaml \
        --ckpt 'serverB/results/.../checkpoint-epoch373.pth' \
        --metadata drafts/figures/anchor_pair_candidates.json \
        --layer-img 23 --layer-txt 24 --gpu 0 \
        --out-dir drafts/figures/pair_grids
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
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--metadata", required=True,
                   help="JSON from anchor_pair_finder.py")
    p.add_argument("--layer-img", type=int, default=23)
    p.add_argument("--layer-txt", type=int, default=24)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-pairs", type=int, default=None,
                   help="Limit to first N pairs (default: all)")
    p.add_argument("--out-dir", default="drafts/figures/pair_grids")
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

    with open(args.metadata) as f:
        meta = json.load(f)
    if args.n_pairs is not None:
        meta = meta[: args.n_pairs]
    print(f"[meta] rendering {len(meta)} pairs")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = {}

    def get_sample(path, caption):
        key = (path, caption)
        if key in cache:
            return cache[key]
        pil = Image.open(path).convert("RGB")
        img_attn, _ = get_image_anchor_attention(
            pil, vision_model, image_transform, alignment_image,
            args.layer_img, device,
        )
        txt_attn, _, token_strs, _ = get_text_anchor_attention(
            caption, tokenizer, language_model, alignment_text,
            args.layer_txt, device,
        )
        cache[key] = (pil, img_attn, txt_attn, token_strs)
        return cache[key]

    t0 = time.time()
    for r, p in enumerate(meta):
        pil_i, ia_i, ta_i, tok_i = get_sample(p["path_i"], p["caption_i"])
        pil_j, ia_j, ta_j, tok_j = get_sample(p["path_j"], p["caption_j"])

        pair_samples = [
            {"image": pil_i, "caption": p["caption_i"],
             "img_attn": ia_i, "txt_attn": ta_i,
             "token_strs": tok_i, "anchor_indices": p["anchors"]},
            {"image": pil_j, "caption": p["caption_j"],
             "img_attn": ia_j, "txt_attn": ta_j,
             "token_strs": tok_j, "anchor_indices": p["anchors"]},
        ]
        anchor_str = "-".join(str(a) for a in p["anchors"])
        out_path = out_dir / (
            f"pair_{r + 1:02d}_score{p['score']:.3f}"
            f"_idx{p['image_idx_i']}-{p['image_idx_j']}"
            f"_a{anchor_str}.png"
        )
        plot_cross_modal_grid(pair_samples, str(out_path))

    print(f"[done] {len(meta)} pairs in {time.time() - t0:.1f}s")
    print(f"[out]  {out_dir}/")


if __name__ == "__main__":
    main()
