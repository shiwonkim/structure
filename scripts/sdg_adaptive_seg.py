"""Softmax Discrimination Gain (SDG) adaptive switching for BA seg.

Calibrates on 20 unlabeled images: measures whether softmax(sim/τ)
improves per-patch class discrimination (top1-top2 margin) vs raw.
If SDG > 1.0, use softmax; otherwise use raw.

Tests on VOC2012 and Pascal Context for both ViT-S and ViT-B.
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

eval_datasets = {
    "voc2012": ("data/pascal_voc", None),
    "pascal_context": ("data/pascal_context", 2000),
}


def compute_sdg(sims_batch, text_feats, tau):
    """Compute Softmax Discrimination Gain on a batch of patch similarities.
    Returns SDG ratio and per-method margins."""
    # Raw path
    raw_feats = F.normalize(sims_batch, dim=-1)
    raw_logits = raw_feats @ text_feats.T  # (N*P, C)
    raw_sorted = raw_logits.sort(dim=-1, descending=True).values
    raw_margin = (raw_sorted[:, 0] - raw_sorted[:, 1]).mean().item()

    # Softmax path
    soft_feats = F.normalize(F.softmax(sims_batch / tau, dim=-1), dim=-1)
    soft_logits = soft_feats @ text_feats.T
    soft_sorted = soft_logits.sort(dim=-1, descending=True).values
    soft_margin = (soft_sorted[:, 0] - soft_sorted[:, 1]).mean().item()

    sdg = soft_margin / raw_margin if raw_margin > 1e-8 else 1.0
    return sdg, raw_margin, soft_margin


def run_seg_eval(vision_model, anchors, text_feats, ds, ds_spec, transform,
                 layer_img, mode, tau=None, max_images=None):
    """Run seg eval."""
    confusion = np.zeros((ds_spec.num_classes, ds_spec.num_classes), dtype=np.int64)
    n = len(ds) if max_images is None else min(max_images, len(ds))
    for i in tqdm(range(n), desc=f"{mode}", leave=False):
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


for ds_name, (data_root, max_imgs) in eval_datasets.items():
    ds, ds_spec = build_dataset(ds_name, data_root=data_root, download=False)
    templates = get_text_templates("ensemble")
    prompt_names = [ds_spec.prompts[c] for c in ds_spec.classes]

    print(f"\n{'#'*60}", flush=True)
    print(f"  DATASET: {ds_name} ({ds_spec.num_classes} classes, {len(ds)} images)", flush=True)
    print(f"{'#'*60}", flush=True)

    for name, m in models.items():
        cfg = load_config(m["config"])
        tau = 0.05

        lvm_name = cfg["alignment"]["lvm_model_name"]
        img_size = int(cfg["features"]["img_size"])
        vm = timm.create_model(lvm_name, pretrained=True, img_size=img_size)
        data_cfg = resolve_data_config(vm.pretrained_cfg, model=vm)
        return_nodes = [f"blocks.{i}.add_1" for i in range(len(vm.blocks))]
        vision_model = create_feature_extractor(vm, return_nodes=return_nodes).float().to(device).eval()
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(data_cfg["mean"], data_cfg["std"]),
        ])

        ckpt = torch.load(m["ckpt"], map_location="cpu", weights_only=False)
        ai = ckpt["alignment_image"].to(device).eval()
        at = ckpt["alignment_text"].to(device).eval()
        anchors = F.normalize(ai.anchors, dim=-1).detach()

        language_model, tokenizer = build_language_encoder(cfg, device)
        text_feats = build_zero_shot_classifier(
            language_model=language_model, tokenizer=tokenizer,
            classnames=prompt_names, templates=templates, dataset=None,
            layer_index=m["layer_txt"], alignment_layer=at,
            num_classes_per_batch=8, device=device,
            pool_txt="none", save_path=None, token_level=True,
        ).to(device)
        text_feats = F.normalize(text_feats.float(), dim=-1)

        print(f"\n  {'='*50}", flush=True)
        print(f"  {name.upper()} × {ds_name}", flush=True)
        print(f"  {'='*50}", flush=True)

        # Calibration: 20 random images
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
                all_sims.append(p_n @ anchors.T)
        sims_batch = torch.cat(all_sims)

        # Compute SDG
        sdg, raw_margin, soft_margin = compute_sdg(sims_batch, text_feats, tau)
        chosen = "softmax" if sdg > 1.0 else "raw"
        print(f"  SDG calibration (20 images):", flush=True)
        print(f"    Raw margin:     {raw_margin:.4f}", flush=True)
        print(f"    Softmax margin: {soft_margin:.4f}", flush=True)
        print(f"    SDG = {sdg:.3f}", flush=True)
        print(f"    → Chosen: {chosen}", flush=True)

        # Evaluate both + report which SDG picked
        miou_raw = run_seg_eval(vision_model, anchors, text_feats, ds, ds_spec,
                                transform, m["layer_img"], "raw", max_images=max_imgs)
        print(f"  raw:     mIoU-fg={100*miou_raw:.2f}%", flush=True)

        miou_soft = run_seg_eval(vision_model, anchors, text_feats, ds, ds_spec,
                                 transform, m["layer_img"], "softmax", tau=tau,
                                 max_images=max_imgs)
        print(f"  softmax: mIoU-fg={100*miou_soft:.2f}%", flush=True)

        actual_best = "softmax" if miou_soft > miou_raw else "raw"
        correct = "CORRECT" if chosen == actual_best else "WRONG"
        print(f"  SDG chose {chosen}, actual best is {actual_best} → {correct}", flush=True)

        del vision_model, language_model
        torch.cuda.empty_cache()

print("\nDONE", flush=True)
