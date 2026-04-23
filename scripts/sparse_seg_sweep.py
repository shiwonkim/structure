"""Sparse similarity seg eval: top-K gating, sparsemax, entmax-1.5.

Tests on VOC2012 for both ViT-S and ViT-B.
"""

import torch, torch.nn.functional as F, os, math, timm
import numpy as np
from timm.data import resolve_data_config
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from entmax import sparsemax, entmax15
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

ds, ds_spec = build_dataset("voc2012", data_root="data/pascal_voc", download=False)
templates = get_text_templates("ensemble")
prompt_names = [ds_spec.prompts[c] for c in ds_spec.classes]


def make_patch_feats(sim, method, tau=0.05, topk=32):
    """Transform raw (P, K) similarities into patch features."""
    if method == "raw":
        return sim
    elif method.startswith("topk_"):
        k = int(method.split("_")[1])
        vals, idx = sim.topk(k, dim=-1)
        out = torch.zeros_like(sim)
        out.scatter_(-1, idx, vals)
        return out
    elif method == "softmax":
        return F.softmax(sim / tau, dim=-1)
    elif method == "sparsemax":
        return sparsemax(sim / tau, dim=-1)
    elif method == "entmax15":
        return entmax15(sim / tau, dim=-1)
    else:
        raise ValueError(method)


methods = [
    "raw",
    "softmax",
    "sparsemax",
    "entmax15",
    "topk_8",
    "topk_16",
    "topk_32",
    "topk_64",
    "topk_128",
    "topk_256",
]

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
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
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

    print(f"\n{'='*60}", flush=True)
    print(f"  {name.upper()} K=512 — VOC2012 ({len(ds)} images)", flush=True)
    print(f"{'='*60}", flush=True)

    results = {}
    for method in methods:
        confusion = np.zeros((ds_spec.num_classes, ds_spec.num_classes), dtype=np.int64)
        for i in tqdm(range(len(ds)), desc=f"{name}/{method}", leave=False):
            pil_img, pil_mask = ds[i]
            img_t = transform(pil_img).unsqueeze(0).float().to(device)
            with torch.no_grad():
                lvm_out = vision_model(img_t)
                feats = list(lvm_out.values())[m["layer_img"]].squeeze(0)
                patches = feats[1:, :]
                p_n = F.normalize(patches, dim=-1)
                sim = p_n @ anchors.T  # (P, K)
                patch_feats = make_patch_feats(sim, method, tau=tau)
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
        results[method] = res["miou_fg"]
        print(f"  {method:20s}  mIoU-fg={100*res['miou_fg']:.2f}%", flush=True)

    print(f"\n  {'Method':<20s} {'mIoU-fg':>10s}", flush=True)
    print(f"  {'-'*30}", flush=True)
    for method, miou in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ***" if miou > results["raw"] else ""
        print(f"  {method:<20s} {100*miou:>9.2f}%{marker}", flush=True)

    del vision_model, language_model
    torch.cuda.empty_cache()

print("\nDONE", flush=True)
