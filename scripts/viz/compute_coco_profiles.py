"""Compute anchor profiles on COCO Karpathy test set and save cache.

Usage:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/viz/compute_coco_profiles.py \
        --gpu 0 --output results/anchor_stats_coco_cache.pkl
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.alignment import *  # noqa
from src.core.src.utils.loader import Loader, merge_dicts
from src.models.text.models import load_llm, load_tokenizer
from timm import create_model
from timm.data import resolve_data_config
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.transforms as transforms


CKPT = "results/alignment-sentence_transformers_all_roberta_large_v1-vit_large_patch14_dinov2.lvd142m-generous-elevator-48/(23, 24)_0.2903/checkpoints/checkpoint-epoch400.pth"
CONFIG = "configs/ba/vitl_roberta/token_k512.yaml"
LAYER_IMG = 23
LAYER_TXT = 24


def _ensure_rgb(img):
    if hasattr(img, "mode") and img.mode != "RGB":
        return img.convert("RGB")
    return img


def load_coco_karpathy_test():
    """Load COCO Karpathy test split (5000 images)."""
    # Karpathy test uses the last 5000 images from val2014
    with open("data/COCO/annotations/captions_val2014.json") as f:
        data = json.load(f)

    # Build image_id -> captions map
    cap_map = {}
    for ann in data["annotations"]:
        iid = ann["image_id"]
        if iid not in cap_map:
            cap_map[iid] = ann["caption"]

    # Karpathy test: last 5000 images sorted by id
    images_sorted = sorted(data["images"], key=lambda x: x["id"])
    test_images = images_sorted[-5000:]

    samples = []
    for img_info in test_images:
        img_id = img_info["id"]
        img_path = f"data/COCO/val2014/{img_info['file_name']}"
        caption = cap_map.get(img_id, "a photo")
        samples.append({"img_id": img_id, "img_path": img_path, "caption": caption})

    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output", default="results/anchor_stats_coco_cache.pkl")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")

    with open(CONFIG) as f:
        cfg = Loader(f).get_single_data()
    cfg = merge_dicts(cfg.get("defaults", {}), cfg.get("overrides", {}))

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
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

    samples = load_coco_karpathy_test()
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"Loaded {len(samples)} COCO Karpathy test samples")

    K = alignment_image.anchors.shape[0]
    N = len(samples)
    a_img = F.normalize(alignment_image.anchors.float(), dim=-1)
    a_txt = F.normalize(alignment_text.anchors.float(), dim=-1)
    tau_img = getattr(alignment_image, "pool_temperature", 0.05)
    tau_txt = getattr(alignment_text, "pool_temperature", 0.05)

    image_profiles = np.zeros((N, K), dtype=np.float32)
    text_profiles = np.zeros((N, K), dtype=np.float32)

    for i, sample in enumerate(tqdm(samples, desc="Computing profiles")):
        # Image
        pil_img = Image.open(sample["img_path"]).convert("RGB")
        img_t = image_transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            lvm_out = vision_model(img_t)
            layer_key = list(lvm_out.keys())[LAYER_IMG]
            feats = lvm_out[layer_key].squeeze(0)
        patches = feats[1:, :].float()
        z_n = F.normalize(patches, dim=-1)
        sim = z_n @ a_img.T
        attn = F.softmax(sim / tau_img, dim=0)
        image_profiles[i] = attn.max(dim=0).values.detach().cpu().numpy()

        # Text
        tokens = tokenizer(sample["caption"], return_tensors="pt",
                           padding=False, truncation=True)
        input_ids = tokens["input_ids"].to(device)
        attn_mask = tokens["attention_mask"].to(device)
        with torch.no_grad():
            out = language_model(input_ids=input_ids, attention_mask=attn_mask)
            hidden = torch.stack(out["hidden_states"]).permute(1, 0, 2, 3)
            text_feats = hidden[0, LAYER_TXT, :, :]
        z_n_t = F.normalize(text_feats.float(), dim=-1)
        sim_t = z_n_t @ a_txt.T
        mask = attn_mask[0].bool()
        logits_t = sim_t / tau_txt
        logits_t[~mask] = float("-inf")
        attn_t = F.softmax(logits_t, dim=0).nan_to_num(0.0)
        text_profiles[i] = attn_t.max(dim=0).values.detach().cpu().numpy()

    print(f"Profiles: image {image_profiles.shape}, text {text_profiles.shape}")

    with open(args.output, "wb") as f:
        pickle.dump({
            "image_profiles": image_profiles,
            "text_profiles": text_profiles,
        }, f)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
