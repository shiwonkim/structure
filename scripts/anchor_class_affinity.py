"""Anchor–class affinity analysis for BridgeAnchors interpretability.

Computes text_class_embeds @ anchors.T to reveal semantic specialization
of learned anchors. Also tests anchor-routing classification as ablation.

Usage:
    PYTHONPATH=. python scripts/anchor_class_affinity.py --gpu 1
"""

import argparse
import os
import sys
import json
import math

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import timm
from timm.data import resolve_data_config
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from pathlib import Path
from collections import defaultdict

from src.evaluation.zero_shot_classifier import build_zero_shot_classifier
from src.models.text.models import load_llm, load_tokenizer
from src.evaluation.zero_shot_segmentation import load_config

IMAGENET_TEMPLATES = [
    "a photo of a {}.",
    "a bad photo of a {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of a {}.",
    "a photo of the small {}.",
]

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CIFAR100_CLASSES = [
    "apple", "aquarium fish", "baby", "bear", "beaver", "bed", "bee",
    "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus",
    "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch",
    "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant",
    "flatfish", "forest", "fox", "girl", "hamster", "house",
    "kangaroo", "keyboard", "lamp", "lawn mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak tree", "orange", "orchid", "otter",
    "palm tree", "pear", "pickup truck", "pine tree", "plain", "plate",
    "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray",
    "road", "rocket", "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider", "squirrel",
    "streetcar", "sunflower", "sweet pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow tree", "wolf",
    "woman", "worm",
]


def load_imagenet_classes(path="data/imagenet/LOC_synset_mapping.txt"):
    classes = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                name = parts[1].split(",")[0].strip()
                classes.append(name)
    return classes


def build_text_embeds(classnames, templates, tokenizer, language_model,
                      alignment_text, layer_txt, device):
    """Build per-class text embeddings via CAP (same as ZS classification)."""
    return build_zero_shot_classifier(
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
    ).to(device)


def build_raw_text_embeds(classnames, templates, tokenizer, language_model,
                          layer_txt, device):
    """Build per-class text embeddings WITHOUT alignment (raw encoder)."""
    return build_zero_shot_classifier(
        language_model=language_model,
        tokenizer=tokenizer,
        classnames=classnames,
        templates=templates,
        dataset=None,
        layer_index=layer_txt,
        alignment_layer=None,
        num_classes_per_batch=8,
        device=device,
        pool_txt="avg",
        save_path=None,
        token_level=False,
    ).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--ckpt",
        default="results/alignment-sentence_transformers_all_MiniLM_L6_v2-"
                "vit_small_patch14_dinov2.lvd142m-wobbly-water-15/"
                "(11, 6)_0.2739/checkpoints/checkpoint-epoch490.pth")
    parser.add_argument("--config", default="configs/ba/vits_minilm/token_k512.yaml")
    parser.add_argument("--outdir", default="results/anchor_affinity")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)

    # Load language model
    print("Loading language model...")
    language_model, tokenizer = None, None
    llm_name = cfg["alignment"]["llm_model_name"]
    language_model = load_llm(llm_name)
    language_model = language_model.float().to(device).eval()
    tokenizer = load_tokenizer(llm_name)

    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    alignment_image = ckpt["alignment_image"].to(device).eval()
    alignment_text = ckpt["alignment_text"].to(device).eval()
    if hasattr(alignment_image, "set_modality"):
        alignment_image.set_modality("image")
    if hasattr(alignment_text, "set_modality"):
        alignment_text.set_modality("text")

    anchors = F.normalize(alignment_image.anchors, dim=-1).detach()  # (K, D)
    K, D = anchors.shape
    print(f"Anchors: K={K}, D={D}")

    layer_txt = 6  # ViT-S + MiniLM

    # ============================================================
    # 1. Anchor-class affinity for multiple class sets
    # ============================================================
    class_sets = {
        "cifar10": CIFAR10_CLASSES,
        "cifar100": CIFAR100_CLASSES,
        "imagenet": load_imagenet_classes(),
    }

    for ds_name, classnames in class_sets.items():
        C = len(classnames)
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} ({C} classes)")
        print(f"{'='*60}")

        # --- Text embeddings via CAP (K-dim profiles) ---
        text_profiles = build_text_embeds(
            classnames, IMAGENET_TEMPLATES, tokenizer, language_model,
            alignment_text, layer_txt, device,
        )  # (C, K)
        text_profiles = F.normalize(text_profiles, dim=-1)

        # --- Raw text embeddings in encoder space (D-dim) ---
        raw_text = build_raw_text_embeds(
            classnames, IMAGENET_TEMPLATES, tokenizer, language_model,
            layer_txt, device,
        )  # (C, D)
        raw_text = F.normalize(raw_text.float().to(device), dim=-1)

        # --- Affinity: raw_text @ anchors.T → (C, K) ---
        affinity = (raw_text @ anchors.T).cpu().numpy()  # (C, K)

        # --- Top-10 anchors per class ---
        print(f"\n--- Top-10 anchors per class (first 20 classes) ---")
        for i in range(min(20, C)):
            top_k_idx = np.argsort(affinity[i])[::-1][:10]
            top_k_val = affinity[i][top_k_idx]
            anchors_str = ", ".join(f"{idx}({val:.3f})" for idx, val in zip(top_k_idx, top_k_val))
            print(f"  {classnames[i]:25s} → [{anchors_str}]")

        # --- Top-10 classes per anchor (first 20 anchors) ---
        print(f"\n--- Top-10 classes per anchor (first 20 anchors) ---")
        for k in range(min(20, K)):
            top_c_idx = np.argsort(affinity[:, k])[::-1][:10]
            top_c_val = affinity[top_c_idx, k]
            classes_str = ", ".join(f"{classnames[c]}({v:.3f})" for c, v in zip(top_c_idx, top_c_val))
            print(f"  Anchor {k:4d} → [{classes_str}]")

        # --- Anchor utilization: how many classes does each anchor serve? ---
        best_anchor_per_class = np.argmax(affinity, axis=1)  # (C,)
        anchor_usage = defaultdict(list)
        for i, a in enumerate(best_anchor_per_class):
            anchor_usage[a].append(classnames[i])

        used_anchors = len(anchor_usage)
        print(f"\n--- Anchor utilization ---")
        print(f"  Unique anchors used (best-per-class): {used_anchors}/{K} "
              f"({100*used_anchors/K:.1f}%)")

        # Top-5 most-used anchors
        sorted_usage = sorted(anchor_usage.items(), key=lambda x: -len(x[1]))
        print(f"  Top-5 most-used anchors:")
        for a, cls_list in sorted_usage[:5]:
            print(f"    Anchor {a}: {len(cls_list)} classes — {cls_list[:8]}")

        # --- Heatmap ---
        if C <= 120:
            fig, ax = plt.subplots(figsize=(max(16, K//20), max(8, C//8)))
            im = ax.imshow(affinity, aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.3)
            ax.set_xlabel("Anchor index")
            ax.set_ylabel("Class")
            ax.set_title(f"Text–Anchor Affinity ({ds_name}, K={K})")
            if C <= 30:
                ax.set_yticks(range(C))
                ax.set_yticklabels(classnames, fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.8)
            plt.tight_layout()
            fig.savefig(outdir / f"affinity_heatmap_{ds_name}.png", dpi=150)
            plt.close(fig)
            print(f"  Saved heatmap → {outdir}/affinity_heatmap_{ds_name}.png")

        # --- Anchor specialization histogram ---
        max_affinity_per_anchor = np.max(affinity, axis=0)  # (K,)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(K), np.sort(max_affinity_per_anchor)[::-1], width=1.0)
        ax.set_xlabel("Anchor (sorted by max class affinity)")
        ax.set_ylabel("Max affinity to any class")
        ax.set_title(f"Anchor specialization ({ds_name})")
        plt.tight_layout()
        fig.savefig(outdir / f"anchor_specialization_{ds_name}.png", dpi=150)
        plt.close(fig)

    # ============================================================
    # 2. Anchor-routing classification (method 2 ablation)
    # ============================================================
    print(f"\n{'='*60}")
    print("Anchor-routing classification ablation (CIFAR-10)")
    print(f"{'='*60}")

    # Load CIFAR-10 test set
    from torchvision import datasets
    lvm_name = cfg["alignment"]["lvm_model_name"]
    img_size = int(cfg["features"]["img_size"])

    print("Loading vision model...")
    vm = timm.create_model(lvm_name, pretrained=True, img_size=img_size)
    data_cfg = resolve_data_config(vm.pretrained_cfg, model=vm)
    return_nodes = [f"blocks.{i}.add_1" for i in range(len(vm.blocks))]
    vision_model = create_feature_extractor(vm, return_nodes=return_nodes)
    vision_model = vision_model.float().to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(data_cfg["mean"], data_cfg["std"]),
    ])
    cifar10 = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(cifar10, batch_size=64, num_workers=4)
    layer_img = 11

    # Build class text profiles (K-dim, via CAP)
    text_profiles_c10 = build_text_embeds(
        CIFAR10_CLASSES, IMAGENET_TEMPLATES, tokenizer, language_model,
        alignment_text, layer_txt, device,
    )  # (10, K)
    text_profiles_c10 = F.normalize(text_profiles_c10, dim=-1)

    # Build raw text in encoder space (D-dim)
    raw_text_c10 = build_raw_text_embeds(
        CIFAR10_CLASSES, IMAGENET_TEMPLATES, tokenizer, language_model,
        layer_txt, device,
    )  # (10, D)
    raw_text_c10 = F.normalize(raw_text_c10.float().to(device), dim=-1)

    # Precompute class-anchor affinity for routing
    class_anchor_affinity = raw_text_c10 @ anchors.T  # (10, K)

    pool_temperature = cfg["training"].get("alignment_layer_kwargs", {}).get("pool_temperature", 0.05)

    correct_cap = 0
    correct_routing = 0
    correct_topk_routing = {k: 0 for k in [32, 64, 128, 256]}
    total = 0

    print("Evaluating...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.float().to(device)
            B = images.shape[0]

            # Extract features
            lvm_out = vision_model(images)
            layer_key = list(lvm_out.keys())[layer_img]
            feats = lvm_out[layer_key]  # (B, T, D)

            # --- Method A: Standard CAP (current pipeline) ---
            img_profiles = alignment_image(feats)  # (B, K)
            img_profiles = F.normalize(img_profiles, dim=-1)
            logits_cap = img_profiles @ text_profiles_c10.T  # (B, 10)
            preds_cap = logits_cap.argmax(dim=-1).cpu()
            correct_cap += (preds_cap == labels).sum().item()

            # --- Method B: Anchor routing (no CAP, just anchor mediation) ---
            # Image: mean of patch-anchor cosine similarities
            patches = F.normalize(feats[:, 1:, :], dim=-1)  # (B, P, D)
            patch_anchor_sim = patches @ anchors.T  # (B, P, K)
            img_anchor_profile = patch_anchor_sim.mean(dim=1)  # (B, K)
            img_anchor_profile = F.normalize(img_anchor_profile, dim=-1)
            # Class scores via anchor affinity
            logits_routing = img_anchor_profile @ class_anchor_affinity.T  # (B, 10)
            preds_routing = logits_routing.argmax(dim=-1).cpu()
            correct_routing += (preds_routing == labels).sum().item()

            # --- Method C: Top-K anchor routing ---
            for topk in correct_topk_routing:
                # Keep only top-K anchors per class
                _, top_idx = class_anchor_affinity.topk(topk, dim=-1)  # (10, topk)
                mask = torch.zeros_like(class_anchor_affinity)
                mask.scatter_(1, top_idx, 1.0)
                masked_affinity = class_anchor_affinity * mask  # (10, K)
                masked_affinity = F.normalize(masked_affinity, dim=-1)
                logits_topk = img_anchor_profile @ masked_affinity.T
                preds_topk = logits_topk.argmax(dim=-1).cpu()
                correct_topk_routing[topk] += (preds_topk == labels).sum().item()

            total += B

    print(f"\nCIFAR-10 Classification Results (N={total}):")
    print(f"  {'Method':<35s} {'Accuracy':>10s}")
    print(f"  {'-'*45}")
    print(f"  {'CAP (standard pipeline)':<35s} {100*correct_cap/total:>9.2f}%")
    print(f"  {'Anchor routing (all K)':<35s} {100*correct_routing/total:>9.2f}%")
    for topk, corr in correct_topk_routing.items():
        print(f"  {f'Anchor routing (top-{topk})':<35s} {100*corr/total:>9.2f}%")

    # Save summary
    summary = {
        "checkpoint": args.ckpt,
        "K": K, "D": D,
        "cifar10_cap_acc": correct_cap / total,
        "cifar10_routing_acc": correct_routing / total,
        "cifar10_topk_routing_acc": {str(k): v/total for k, v in correct_topk_routing.items()},
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary → {outdir}/summary.json")


if __name__ == "__main__":
    main()
