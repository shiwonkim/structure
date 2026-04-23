"""Entropy-matched segmentation: calibrate τ per dataset to match
text-side entropy, with adaptive raw fallback.

For each (model, dataset), computes text profile entropy, then finds
τ* via binary search on a small image batch such that image patch
entropy ≈ text entropy. If no τ in [0.001, 10] matches (text entropy
is too high), falls back to raw similarity.

Tests on VOC2012 for ViT-S and ViT-B.
"""

import torch, torch.nn.functional as F, os, math, timm
import numpy as np
from timm.data import resolve_data_config
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from src.evaluation.zero_shot_classifier import build_zero_shot_classifier
from src.evaluation.zero_shot_segmentation import (
    load_config, build_dataset, build_language_encoder, get_text_templates,
    update_confusion_matrix, compute_iou_from_confusion,
)
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0")

models = {
    "vits": {
        "config": "configs/ba/vits_minilm/token_k512.yaml",
        "ckpt": "results/alignment-sentence_transformers_all_MiniLM_L6_v2-"
                "vit_small_patch14_dinov2.lvd142m-wobbly-water-15/"
                "(11, 6)_0.2739/checkpoints/checkpoint-epoch490.pth",
        "layer_img": 11, "layer_txt": 6,
    },
    "vitb": {
        "config": "configs/ba/vitb_mpnet/token_k512.yaml",
        "ckpt": "results/alignment-sentence_transformers_all_mpnet_base_v2-"
                "vit_base_patch14_dinov2.lvd142m-rich-flower-83/"
                "(11, 12)_0.2852/checkpoints/checkpoint-epoch408.pth",
        "layer_img": 11, "layer_txt": 12,
    },
}


def entropy(probs, dim=-1):
    """Shannon entropy along a dimension."""
    return -(probs * torch.log(probs + 1e-10)).sum(dim=dim)


def patch_entropy_at_tau(sims_batch, tau):
    """Mean per-patch entropy of softmax(sim/τ, dim=-1) over a batch of images."""
    probs = F.softmax(sims_batch / tau, dim=-1)
    return entropy(probs, dim=-1).mean().item()


def raw_effective_entropy(sims_batch):
    """'Entropy' of L2-normalized raw similarity treated as a quasi-distribution.
    After L2 norm + shift to positive + renormalize, compute entropy."""
    normed = F.normalize(sims_batch, dim=-1)  # unit vectors
    # Shift to [0, 1] range per patch, then normalize to sum to 1
    shifted = normed - normed.min(dim=-1, keepdim=True).values
    probs = shifted / shifted.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    return entropy(probs, dim=-1).mean().item()


def calibrate_tau(sims_batch, target_entropy, tau_low=0.001, tau_high=10.0,
                  n_steps=30):
    """Binary search for τ* such that patch entropy ≈ target_entropy."""
    for _ in range(n_steps):
        tau_mid = (tau_low + tau_high) / 2
        ent = patch_entropy_at_tau(sims_batch, tau_mid)
        if ent < target_entropy:
            tau_low = tau_mid  # need softer → higher τ
        else:
            tau_high = tau_mid  # need sharper → lower τ
    return (tau_low + tau_high) / 2


def run_seg_eval(vision_model, anchors, text_feats, ds, ds_spec, transform,
                 layer_img, mode, tau=None, device=device):
    """Run full VOC seg eval with given mode ('raw' or 'softmax')."""
    confusion = np.zeros((ds_spec.num_classes, ds_spec.num_classes), dtype=np.int64)
    for i in tqdm(range(len(ds)), desc=f"{mode}", leave=False):
        pil_img, pil_mask = ds[i]
        img_t = transform(pil_img).unsqueeze(0).float().to(device)
        with torch.no_grad():
            lvm_out = vision_model(img_t)
            feats = list(lvm_out.values())[layer_img].squeeze(0)
            patches = feats[1:, :]
            p_n = F.normalize(patches, dim=-1)
            sim = p_n @ anchors.T
            if mode == "softmax":
                patch_feats = F.softmax(sim / tau, dim=-1)
            else:
                patch_feats = sim
            patch_feats = F.normalize(patch_feats, dim=-1)
        sim_map = patch_feats @ text_feats.T
        P = sim_map.shape[0]; h = int(round(math.sqrt(P)))
        sim_map = sim_map.view(h, h, ds_spec.num_classes).permute(2, 0, 1).unsqueeze(0)
        gt = np.array(pil_mask); H, W = gt.shape
        sim_up = F.interpolate(sim_map.float(), size=(H, W), mode="bilinear", align_corners=False)
        pred = sim_up.squeeze(0).argmax(dim=0).cpu().numpy().astype(np.int64)
        update_confusion_matrix(confusion, gt, pred,
            num_classes=ds_spec.num_classes, ignore_index=ds_spec.ignore_index)
    res = compute_iou_from_confusion(confusion, exclude_background=ds_spec.has_background)
    return res["miou_fg"]


ds, ds_spec = build_dataset("voc2012", data_root="data/pascal_voc", download=False)
templates = get_text_templates("ensemble")
prompt_names = [ds_spec.prompts[c] for c in ds_spec.classes]

for name, m in models.items():
    cfg = load_config(m["config"])
    lvm_name = cfg["alignment"]["lvm_model_name"]
    img_size = int(cfg["features"]["img_size"])
    vm = timm.create_model(lvm_name, pretrained=True, img_size=img_size)
    data_cfg = resolve_data_config(vm.pretrained_cfg, model=vm)
    return_nodes = [f"blocks.{i}.add_1" for i in range(len(vm.blocks))]
    vision_model = create_feature_extractor(vm, return_nodes=return_nodes).float().to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(data_cfg["mean"], data_cfg["std"]),
    ])

    ckpt = torch.load(m["ckpt"], map_location="cpu", weights_only=False)
    ai = ckpt["alignment_image"].to(device).eval()
    at = ckpt["alignment_text"].to(device).eval()
    anchors = F.normalize(ai.anchors, dim=-1).detach()
    K = anchors.shape[0]

    language_model, tokenizer = build_language_encoder(cfg, device)
    text_feats = build_zero_shot_classifier(
        language_model=language_model, tokenizer=tokenizer,
        classnames=prompt_names, templates=templates, dataset=None,
        layer_index=m["layer_txt"], alignment_layer=at,
        num_classes_per_batch=8, device=device,
        pool_txt="none", save_path=None, token_level=True,
    ).to(device)
    text_feats = F.normalize(text_feats.float(), dim=-1)  # (C, K)

    print(f"\n{'='*60}", flush=True)
    print(f"  {name.upper()} K={K} — Entropy Matching — VOC2012", flush=True)
    print(f"{'='*60}", flush=True)

    # Step 1: Text profile entropy
    # Text profiles are already L2-normalized K-dim vectors.
    # To compute meaningful entropy, softmax-normalize them first.
    # But they're CAP outputs (already went through softmax+weighted sum+L2 norm).
    # Treat absolute values as pseudo-probabilities:
    text_abs = text_feats.abs()
    text_probs = text_abs / text_abs.sum(dim=-1, keepdim=True)
    text_ent = entropy(text_probs, dim=-1)
    mean_text_ent = text_ent.mean().item()
    max_ent = math.log(K)
    print(f"  Text profile entropy: {mean_text_ent:.3f} / {max_ent:.3f} "
          f"({100*mean_text_ent/max_ent:.1f}%)", flush=True)
    print(f"  Per-class text entropy range: [{text_ent.min():.3f}, {text_ent.max():.3f}]", flush=True)

    # Step 2: Collect image patch similarities from 20 random images
    np.random.seed(42)
    cal_indices = np.random.choice(len(ds), 20, replace=False)
    all_sims = []
    with torch.no_grad():
        for idx in cal_indices:
            pil_img, _ = ds[idx]
            img_t = transform(pil_img).unsqueeze(0).float().to(device)
            lvm_out = vision_model(img_t)
            feats = list(lvm_out.values())[m["layer_img"]].squeeze(0)
            patches = feats[1:, :]
            p_n = F.normalize(patches, dim=-1)
            sim = p_n @ anchors.T  # (P, K)
            all_sims.append(sim)
    sims_batch = torch.cat(all_sims)  # (N*P, K)
    print(f"  Calibration: {sims_batch.shape[0]} patches from {len(cal_indices)} images", flush=True)

    # Step 3: Raw similarity effective entropy
    raw_ent = raw_effective_entropy(sims_batch)
    print(f"  Raw sim effective entropy: {raw_ent:.3f} ({100*raw_ent/max_ent:.1f}%)", flush=True)

    # Step 4: Binary search for τ*
    tau_star = calibrate_tau(sims_batch, mean_text_ent)
    softmax_ent = patch_entropy_at_tau(sims_batch, tau_star)
    print(f"  Calibrated τ* = {tau_star:.4f}", flush=True)
    print(f"  Softmax entropy at τ*: {softmax_ent:.3f} (target: {mean_text_ent:.3f})", flush=True)

    # Step 5: Decide raw vs softmax
    raw_dist = abs(raw_ent - mean_text_ent)
    softmax_dist = abs(softmax_ent - mean_text_ent)
    chosen = "raw" if raw_dist < softmax_dist else "softmax"
    print(f"  Distance to text entropy: raw={raw_dist:.3f}, softmax={softmax_dist:.3f}", flush=True)
    print(f"  → Chosen: {chosen}" + (f" (τ*={tau_star:.4f})" if chosen == "softmax" else ""), flush=True)

    # Step 6: Evaluate all three: raw, softmax@τ*, and the chosen one
    print(f"\n  Evaluating...", flush=True)

    miou_raw = run_seg_eval(vision_model, anchors, text_feats, ds, ds_spec,
                            transform, m["layer_img"], "raw", device=device)
    print(f"  raw:              mIoU-fg={100*miou_raw:.2f}%", flush=True)

    miou_softmax = run_seg_eval(vision_model, anchors, text_feats, ds, ds_spec,
                                transform, m["layer_img"], "softmax", tau=tau_star, device=device)
    print(f"  softmax(τ*={tau_star:.4f}): mIoU-fg={100*miou_softmax:.2f}%", flush=True)

    # Also try training τ for reference
    miou_train_tau = run_seg_eval(vision_model, anchors, text_feats, ds, ds_spec,
                                  transform, m["layer_img"], "softmax", tau=0.05, device=device)
    print(f"  softmax(τ=0.05):  mIoU-fg={100*miou_train_tau:.2f}%", flush=True)

    print(f"\n  Summary:", flush=True)
    print(f"    Entropy matching chose: {chosen}", flush=True)
    best = max(miou_raw, miou_softmax, miou_train_tau)
    for label, val in [("raw", miou_raw), (f"softmax(τ*={tau_star:.4f})", miou_softmax),
                        ("softmax(τ=0.05)", miou_train_tau)]:
        marker = " ← best" if val == best else ""
        em = " ← entropy-matched choice" if (chosen == "raw" and "raw" == label) or \
             (chosen == "softmax" and f"τ*=" in label) else ""
        print(f"    {label:30s}  {100*val:.2f}%{marker}{em}", flush=True)

    del vision_model, language_model
    torch.cuda.empty_cache()

print("\nDONE", flush=True)
