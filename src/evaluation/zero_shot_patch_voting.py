"""Patch-level voting for zero-shot classification using BA anchor codebook.

Instead of pooling all patches into a single CAP profile then classifying,
each patch independently votes for a class via its anchor similarity profile.
This is the classification analogue of the segmentation anchor_codebook method.

Usage:
    python src/evaluation/zero_shot_patch_voting.py \
        --config configs/ba/vitl_roberta/token_k512.yaml \
        --checkpoint results/alignment-.../checkpoint-epoch400.pth \
        --layer-img 23 --layer-txt 24 \
        --datasets cifar10,cifar100,stl10,food101,imagenet,dtd \
        --pool-modes mean,max,topk16 \
        --gpu 0
"""
import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
import torchmetrics
import torchvision.transforms as transforms
from loguru import logger
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor
from timm import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.alignment import *  # noqa: register all alignment layers
from src.alignment.alignment_factory import AlignmentFactory
from src.core.src.utils.loader import Loader, merge_dicts
from src.dataset_preparation.data_utils import get_datasets
from src.evaluation.zero_shot_classifier import build_zero_shot_classifier
from src.evaluation.consts import DATASETS_TO_CLASSES, DATASETS_TO_TEMPLATES
from src.models.text.models import load_llm, load_tokenizer


SIMPLE_TEMPLATE = ["{}"]


def _ensure_rgb_image(img):
    from PIL import Image
    if isinstance(img, Image.Image) and img.mode != "RGB":
        return img.convert("RGB")
    return img


def load_checkpoint(config, ckpt_path, device):
    """Load alignment layers from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    alignment_image = ckpt["alignment_image"]
    alignment_text = ckpt["alignment_text"]

    if hasattr(alignment_image, "set_modality"):
        alignment_image.set_modality("image")
    if hasattr(alignment_text, "set_modality"):
        alignment_text.set_modality("text")

    alignment_image.eval().to(device)
    alignment_text.eval().to(device)

    return alignment_image, alignment_text


def patch_vote_classify(
    patch_profiles: torch.Tensor,
    class_profiles: torch.Tensor,
    pool_mode: str = "mean",
) -> torch.Tensor:
    """Classify by per-patch voting.

    Args:
        patch_profiles: (P, K) per-patch anchor similarity profiles
        class_profiles: (C, K) per-class CAP profiles (L2-normalized)
        pool_mode: "mean", "max", "topkN" (e.g. "topk16")

    Returns:
        logits: (C,) classification logits
    """
    # Normalize patch profiles
    patch_profiles = F.normalize(patch_profiles, dim=-1)

    # Per-patch similarity to each class: (P, C)
    sim = patch_profiles @ class_profiles.T

    if pool_mode == "mean":
        return sim.mean(dim=0)
    elif pool_mode == "max":
        return sim.max(dim=0).values
    elif pool_mode.startswith("topk"):
        k = int(pool_mode[4:])
        k = min(k, sim.shape[0])
        topk_vals, _ = sim.topk(k, dim=0)
        return topk_vals.mean(dim=0)
    elif pool_mode == "softmax":
        # Soft attention pooling: weight patches by their max class similarity
        weights = sim.max(dim=1).values  # (P,)
        weights = F.softmax(weights / 0.1, dim=0)  # temperature-scaled
        return (sim * weights.unsqueeze(-1)).sum(dim=0)
    else:
        raise ValueError(f"Unknown pool_mode: {pool_mode}")


def evaluate_dataset(
    dataset_name: str,
    vision_model,
    image_transform,
    alignment_image,
    alignment_text,
    language_model,
    tokenizer,
    layer_img: int,
    layer_txt: int,
    pool_modes: List[str],
    device: torch.device,
    batch_size: int = 256,
    use_extended_prompts: bool = True,
) -> Dict[str, float]:
    """Evaluate patch-voting classification on one dataset."""

    classnames = DATASETS_TO_CLASSES[dataset_name.lower()]
    templates = (
        DATASETS_TO_TEMPLATES.get(dataset_name.lower(), SIMPLE_TEMPLATE)
        if use_extended_prompts
        else SIMPLE_TEMPLATE
    )

    # Build class profiles via CAP (same as standard zero-shot)
    class_profiles = build_zero_shot_classifier(
        language_model=language_model,
        tokenizer=tokenizer,
        classnames=classnames,
        templates=templates,
        dataset=None,
        layer_index=layer_txt,
        alignment_layer=alignment_text,
        num_classes_per_batch=8,
        device=device,
        pool_txt="none",
        save_path=None,
        token_level=True,
    ).float().to(device)  # (C, K)

    # Also build standard CAP classifier for comparison
    # (this is the default zero-shot inference)

    # Get dataset
    _, eval_dataset = get_datasets(dataset_name, transform=image_transform, root_dir="data/")

    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, drop_last=False, pin_memory=False,
    )

    # Get anchor matrix for patch-level codebook
    anchors = alignment_image.anchors.data  # (K, D)

    # Metrics per pool mode + standard CAP
    all_modes = pool_modes + ["cap"]
    metrics = {
        mode: torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=len(classnames), top_k=1, average="micro"
        )
        for mode in all_modes
    }

    from tqdm import tqdm
    a_n = F.normalize(anchors, dim=-1)  # (K, D)

    for batch in tqdm(eval_loader, desc=dataset_name, file=sys.stdout):
        if len(batch) == 2:
            images, targets = batch
        elif len(batch) == 3:
            images, _, targets = batch
        else:
            continue

        images = images.to(device)

        with torch.no_grad():
            lvm_out = vision_model(images)
            layer_key = list(lvm_out.keys())[layer_img]
            feats = lvm_out[layer_key]  # (B, T, D)

            # Standard CAP path for comparison
            cap_profiles = alignment_image(feats)  # (B, K)
            cap_profiles = F.normalize(cap_profiles, dim=-1)
            cap_logits = 100.0 * cap_profiles @ class_profiles.T  # (B, C)
            metrics["cap"].update(cap_logits.cpu(), targets)

            # Batch patch-level voting: (B, P, D) → (B, P, K)
            patches = feats[:, 1:, :]  # (B, P, D), strip CLS
            z_n = F.normalize(patches, dim=-1)
            all_patch_profiles = z_n @ a_n.T  # (B, P, K)
            all_patch_profiles = F.normalize(all_patch_profiles, dim=-1)

            # Per-patch similarity to classes: (B, P, C)
            patch_class_sim = all_patch_profiles @ class_profiles.T

            for mode in pool_modes:
                if mode == "mean":
                    logits = patch_class_sim.mean(dim=1)  # (B, C)
                elif mode == "max":
                    logits = patch_class_sim.max(dim=1).values
                elif mode.startswith("topk"):
                    k = min(int(mode[4:]), patch_class_sim.shape[1])
                    topk_vals, _ = patch_class_sim.topk(k, dim=1)
                    logits = topk_vals.mean(dim=1)
                elif mode == "softmax":
                    weights = patch_class_sim.max(dim=2).values  # (B, P)
                    weights = F.softmax(weights / 0.1, dim=1).unsqueeze(-1)
                    logits = (patch_class_sim * weights).sum(dim=1)
                else:
                    continue
                metrics[mode].update((100.0 * logits).cpu(), targets)

    results = {}
    for mode in all_modes:
        acc = metrics[mode].compute().item() * 100
        results[mode] = acc

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--layer-img", type=int, required=True)
    p.add_argument("--layer-txt", type=int, required=True)
    p.add_argument("--datasets", default="cifar10,cifar100,stl10,food101,imagenet,dtd")
    p.add_argument("--pool-modes", default="mean,max,topk16,topk32,softmax")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load config
    with open(args.config) as f:
        cfg = Loader(f).get_single_data()
    cfg = merge_dicts(cfg.get("defaults", {}), cfg.get("overrides", {}))

    # Load models
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
        transforms.Lambda(_ensure_rgb_image),
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config["mean"], std=data_config["std"]),
    ])

    language_model = load_llm(llm_name).to(device)
    tokenizer = load_tokenizer(llm_name)

    alignment_image, alignment_text = load_checkpoint(cfg, args.checkpoint, device)

    datasets = [d.strip() for d in args.datasets.split(",")]
    pool_modes = [m.strip() for m in args.pool_modes.split(",")]

    print(f"\n{'Dataset':>12} |", end="")
    for mode in pool_modes + ["cap"]:
        print(f" {mode:>8} |", end="")
    print()
    print("-" * (14 + 11 * (len(pool_modes) + 1)))

    for ds in datasets:
        try:
            results = evaluate_dataset(
                dataset_name=ds,
                vision_model=vision_model,
                image_transform=image_transform,
                alignment_image=alignment_image,
                alignment_text=alignment_text,
                language_model=language_model,
                tokenizer=tokenizer,
                layer_img=args.layer_img,
                layer_txt=args.layer_txt,
                pool_modes=pool_modes,
                device=device,
                batch_size=args.batch_size,
            )
            print(f"{ds:>12} |", end="")
            for mode in pool_modes + ["cap"]:
                print(f" {results[mode]:7.1f}% |", end="")
            print()
        except Exception as e:
            print(f"{ds:>12} | FAILED: {e}")

    print()


if __name__ == "__main__":
    main()
