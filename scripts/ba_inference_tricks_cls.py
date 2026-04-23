"""BA inference tricks for zero-shot classification.

Tests several inference-time enhancements for BridgeAnchors that require
no retraining — all operate on the same checkpoint.

Tricks implemented:
  1. Multi-scale CAP: average profiles from multiple temperatures
  2. CLS+CAP ensemble: weighted combination of CLS profile and CAP profile
  3. Anchor subset selection: mask low-affinity anchors per class
  4. Patch filtering via CLS attention: drop background patches before CAP
  5. CLS+CAP concat: concatenate CLS and CAP profiles (2K-dim)

Usage:
    PYTHONPATH=. python scripts/ba_inference_tricks_cls.py --gpu 1
"""

import argparse
import os
import json
import math

import torch
import torch.nn.functional as F
import numpy as np
import timm
from timm.data import resolve_data_config
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import datasets, transforms
from pathlib import Path

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

DATASET_CLASSES = {
    "cifar10": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ],
    "cifar100": [
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
    ],
    "stl10": [
        "airplane", "bird", "car", "cat", "deer",
        "dog", "horse", "monkey", "ship", "truck",
    ],
}


# ------------------------------------------------------------------
# Inference methods — each takes (feats, anchors, tau, ...) and
# returns a (B, K) or (B, 2K) profile
# ------------------------------------------------------------------

def cap_profile(feats, anchors, tau):
    """Standard CAP: softmax(patch·anchor/τ) attention → weighted sim."""
    patches = F.normalize(feats[:, 1:, :], dim=-1)  # (B, P, D)
    a_n = F.normalize(anchors, dim=-1)               # (K, D)
    sim = patches @ a_n.T                             # (B, P, K)
    attn = F.softmax(sim / tau, dim=1)                # (B, P, K)
    profile = (attn * sim).sum(dim=1)                 # (B, K)
    return F.normalize(profile, dim=-1)


def cls_profile(feats, anchors):
    """CLS-only profile: CLS_token · anchors → normalize."""
    cls = F.normalize(feats[:, 0, :], dim=-1)  # (B, D)
    a_n = F.normalize(anchors, dim=-1)          # (K, D)
    profile = cls @ a_n.T                       # (B, K)
    return F.normalize(profile, dim=-1)


def multiscale_cap(feats, anchors, taus=(0.05, 0.03, 0.01)):
    """Average CAP profiles across multiple temperatures."""
    profiles = [cap_profile(feats, anchors, t) for t in taus]
    avg = torch.stack(profiles).mean(dim=0)  # (B, K)
    return F.normalize(avg, dim=-1)


def cls_cap_ensemble(feats, anchors, tau, alpha=0.5):
    """Weighted average of CLS and CAP profiles."""
    cap = cap_profile(feats, anchors, tau)
    cls = cls_profile(feats, anchors)
    combined = alpha * cls + (1 - alpha) * cap
    return F.normalize(combined, dim=-1)


def cls_cap_concat(feats, anchors, tau):
    """Concatenate CLS and CAP profiles → 2K-dim descriptor."""
    cap = cap_profile(feats, anchors, tau)
    cls = cls_profile(feats, anchors)
    return torch.cat([cls, cap], dim=-1)  # (B, 2K) — normalized per-half


def anchor_subset_classify(img_profiles, text_profiles, class_anchor_affinity,
                           topk=64):
    """Per-class anchor masking: keep only top-K anchors per class."""
    K = img_profiles.shape[-1]
    _, top_idx = class_anchor_affinity.topk(topk, dim=-1)  # (C, topk)
    mask = torch.zeros_like(class_anchor_affinity)
    mask.scatter_(1, top_idx, 1.0)  # (C, K)

    # Masked cosine: zero out irrelevant anchors per class
    masked_text = text_profiles * mask  # (C, K)
    masked_text = F.normalize(masked_text, dim=-1)
    return img_profiles @ masked_text.T  # (B, C)


def patch_filtered_cap(feats, anchors, tau, keep_ratio=0.5):
    """Filter patches by CLS attention before CAP.
    Uses self-attention from the CLS token to identify salient patches."""
    B, T, D = feats.shape
    P = T - 1
    keep = max(1, int(P * keep_ratio))

    cls = feats[:, 0:1, :]          # (B, 1, D)
    patches = feats[:, 1:, :]       # (B, P, D)

    # CLS-patch attention as saliency
    cls_n = F.normalize(cls, dim=-1)
    patches_n = F.normalize(patches, dim=-1)
    attn_scores = (cls_n @ patches_n.transpose(-1, -2)).squeeze(1)  # (B, P)

    # Keep top-k patches
    _, top_idx = attn_scores.topk(keep, dim=-1)  # (B, keep)
    top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, D)
    filtered = torch.gather(patches, 1, top_idx_exp)  # (B, keep, D)

    # CAP on filtered patches
    f_n = F.normalize(filtered, dim=-1)
    a_n = F.normalize(anchors, dim=-1)
    sim = f_n @ a_n.T                    # (B, keep, K)
    attn = F.softmax(sim / tau, dim=1)   # (B, keep, K)
    profile = (attn * sim).sum(dim=1)    # (B, K)
    return F.normalize(profile, dim=-1)


# ------------------------------------------------------------------
# Evaluation harness
# ------------------------------------------------------------------

def evaluate_method(method_fn, vision_model, alignment_image, layer_img,
                    text_classifier, loader, device, **kwargs):
    """Run a classification method and return accuracy."""
    anchors = alignment_image.anchors.detach()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.float().to(device)
            B = images.shape[0]

            lvm_out = vision_model(images)
            layer_key = list(lvm_out.keys())[layer_img]
            feats = lvm_out[layer_key]  # (B, T, D)

            profiles = method_fn(feats, anchors, **kwargs)  # (B, K) or (B, 2K)
            profiles = F.normalize(profiles, dim=-1)
            logits = profiles @ text_classifier.T  # (B, C)
            preds = logits.argmax(dim=-1).cpu()
            correct += (preds == labels).sum().item()
            total += B
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--ckpt",
        default="results/alignment-sentence_transformers_all_MiniLM_L6_v2-"
                "vit_small_patch14_dinov2.lvd142m-wobbly-water-15/"
                "(11, 6)_0.2739/checkpoints/checkpoint-epoch490.pth")
    parser.add_argument("--config", default="configs/ba/vits_minilm/token_k512.yaml")
    parser.add_argument("--outdir", default="results/ba_cls_tricks")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    tau = cfg["training"].get("alignment_layer_kwargs", {}).get("pool_temperature", 0.05)

    # Load models
    print("Loading language model...")
    llm_name = cfg["alignment"]["llm_model_name"]
    language_model = load_llm(llm_name)
    language_model = language_model.float().to(device).eval()
    tokenizer = load_tokenizer(llm_name)

    print("Loading checkpoint...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    alignment_image = ckpt["alignment_image"].to(device).eval()
    alignment_text = ckpt["alignment_text"].to(device).eval()
    if hasattr(alignment_image, "set_modality"):
        alignment_image.set_modality("image")
    if hasattr(alignment_text, "set_modality"):
        alignment_text.set_modality("text")

    anchors = F.normalize(alignment_image.anchors, dim=-1).detach()
    K = anchors.shape[0]
    layer_img = 11
    layer_txt = 6

    print("Loading vision model...")
    lvm_name = cfg["alignment"]["lvm_model_name"]
    img_size = int(cfg["features"]["img_size"])
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

    # Results collector
    all_results = {}

    for ds_name in ["cifar10", "cifar100", "stl10"]:
        classnames = DATASET_CLASSES[ds_name]
        C = len(classnames)
        print(f"\n{'='*60}")
        print(f"{ds_name} ({C} classes)")
        print(f"{'='*60}")

        # Load dataset
        if ds_name == "cifar10":
            ds = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
        elif ds_name == "cifar100":
            ds = datasets.CIFAR100(root="data", train=False, download=True, transform=transform)
        elif ds_name == "stl10":
            ds = datasets.STL10(root="data", split="test", download=True, transform=transform)
        loader = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=4)

        # Build text classifier (K-dim profiles via CAP)
        text_profiles = build_zero_shot_classifier(
            language_model=language_model, tokenizer=tokenizer,
            classnames=classnames, templates=IMAGENET_TEMPLATES,
            dataset=None, layer_index=layer_txt,
            alignment_layer=alignment_text,
            num_classes_per_batch=8, device=device,
            pool_txt="none", save_path=None, token_level=True,
        ).to(device)
        text_profiles = F.normalize(text_profiles, dim=-1)

        # Raw text embeds for anchor subset selection
        raw_text = build_zero_shot_classifier(
            language_model=language_model, tokenizer=tokenizer,
            classnames=classnames, templates=IMAGENET_TEMPLATES,
            dataset=None, layer_index=layer_txt,
            alignment_layer=None, num_classes_per_batch=8, device=device,
            pool_txt="avg", save_path=None, token_level=False,
        ).float().to(device)
        raw_text = F.normalize(raw_text, dim=-1)
        class_anchor_affinity = raw_text @ anchors.T  # (C, K)

        # Also build 2K-dim text classifier for concat method
        cls_text = cls_profile(
            # Fake a (C, 1, D)-shaped input won't work — need raw text embeds
            # Instead: raw_text @ anchors.T gives CLS-style profiles
            None, None
        ) if False else None  # placeholder
        # For concat: text side also needs 2K dim
        text_cls_profiles = F.normalize(raw_text @ anchors.T, dim=-1)  # (C, K)
        text_2k = torch.cat([text_cls_profiles, text_profiles], dim=-1)  # (C, 2K)
        text_2k = F.normalize(text_2k, dim=-1)

        results = {}

        # --- 0. Baseline: standard CAP ---
        print("  [0] CAP baseline (τ=0.05)...", end=" ", flush=True)
        acc = evaluate_method(
            lambda f, a, **kw: cap_profile(f, a, tau),
            vision_model, alignment_image, layer_img,
            text_profiles, loader, device,
        )
        results["cap_baseline"] = acc
        print(f"{100*acc:.2f}%")

        # --- 1. Multi-scale CAP ---
        for taus in [(0.05, 0.03), (0.05, 0.03, 0.01), (0.07, 0.05, 0.03)]:
            label = f"multiscale_{'_'.join(str(t) for t in taus)}"
            print(f"  [1] Multi-scale τ={taus}...", end=" ", flush=True)
            acc = evaluate_method(
                lambda f, a, taus=taus, **kw: multiscale_cap(f, a, taus),
                vision_model, alignment_image, layer_img,
                text_profiles, loader, device,
            )
            results[label] = acc
            print(f"{100*acc:.2f}%")

        # --- 2. CLS+CAP ensemble ---
        for alpha in [0.1, 0.2, 0.3, 0.5, 0.7]:
            label = f"cls_cap_alpha{alpha}"
            print(f"  [2] CLS+CAP ensemble α={alpha}...", end=" ", flush=True)
            acc = evaluate_method(
                lambda f, a, alpha=alpha, **kw: cls_cap_ensemble(f, a, tau, alpha),
                vision_model, alignment_image, layer_img,
                text_profiles, loader, device,
            )
            results[label] = acc
            print(f"{100*acc:.2f}%")

        # --- 3. Anchor subset selection ---
        for topk in [32, 64, 128, 256]:
            label = f"anchor_subset_top{topk}"
            print(f"  [3] Anchor subset top-{topk}...", end=" ", flush=True)

            def subset_method(feats, anchors, topk=topk, **kw):
                prof = cap_profile(feats, anchors, tau)
                return prof  # profiles are standard; masking happens at classify time

            # Custom eval: need per-class masking
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels_batch in loader:
                    images = images.float().to(device)
                    lvm_out = vision_model(images)
                    layer_key = list(lvm_out.keys())[layer_img]
                    feats = lvm_out[layer_key]
                    img_prof = cap_profile(feats, anchors, tau)
                    logits = anchor_subset_classify(
                        img_prof, text_profiles, class_anchor_affinity, topk=topk
                    )
                    preds = logits.argmax(dim=-1).cpu()
                    correct += (preds == labels_batch).sum().item()
                    total += labels_batch.shape[0]
            acc = correct / total
            results[label] = acc
            print(f"{100*acc:.2f}%")

        # --- 4. Patch filtering ---
        for keep in [0.25, 0.5, 0.75]:
            label = f"patch_filter_{keep}"
            print(f"  [4] Patch filter keep={keep}...", end=" ", flush=True)
            acc = evaluate_method(
                lambda f, a, keep=keep, **kw: patch_filtered_cap(f, a, tau, keep),
                vision_model, alignment_image, layer_img,
                text_profiles, loader, device,
            )
            results[label] = acc
            print(f"{100*acc:.2f}%")

        # --- 5. CLS+CAP concat (2K-dim) ---
        print("  [5] CLS+CAP concat (2K-dim)...", end=" ", flush=True)
        acc = evaluate_method(
            lambda f, a, **kw: cls_cap_concat(f, a, tau),
            vision_model, alignment_image, layer_img,
            text_2k, loader, device,
        )
        results["cls_cap_concat"] = acc
        print(f"{100*acc:.2f}%")

        # --- Summary ---
        print(f"\n  {'Method':<40s} {'Accuracy':>10s}")
        print(f"  {'-'*50}")
        for method, acc in sorted(results.items(), key=lambda x: -x[1]):
            marker = " ***" if acc > results["cap_baseline"] else ""
            print(f"  {method:<40s} {100*acc:>9.2f}%{marker}")

        all_results[ds_name] = results

    # Save
    with open(outdir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {outdir}/results.json")


if __name__ == "__main__":
    main()
