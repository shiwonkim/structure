"""Analyze per-patch similarity distributions before/after softmax(dim=-1).

For a few VOC images, show how raw similarities and softmax-transformed
similarities differ between ViT-S and ViT-B, and how this affects the
downstream argmax prediction.
"""

import torch, torch.nn.functional as F, os, math, timm
import numpy as np
from timm.data import resolve_data_config
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from src.evaluation.zero_shot_classifier import build_zero_shot_classifier
from src.evaluation.zero_shot_segmentation import (
    load_config, build_dataset, build_language_encoder, get_text_templates,
)
from src.models.text.models import load_llm, load_tokenizer

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

np.random.seed(42)
sample_indices = np.random.choice(len(ds), 5, replace=False)

for name, m in models.items():
    cfg = load_config(m["config"])
    tau = 0.05

    print(f"\n{'='*70}", flush=True)
    print(f"  {name.upper()} — K=512, τ={tau}", flush=True)
    print(f"{'='*70}", flush=True)

    # Load vision model
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

    # Load checkpoint
    ckpt = torch.load(m["ckpt"], map_location="cpu", weights_only=False)
    ai = ckpt["alignment_image"].to(device).eval()
    at = ckpt["alignment_text"].to(device).eval()
    anchors = F.normalize(ai.anchors, dim=-1).detach()
    K = anchors.shape[0]

    # Load language model + build text classifier
    language_model, tokenizer = build_language_encoder(cfg, device)
    text_feats = build_zero_shot_classifier(
        language_model=language_model, tokenizer=tokenizer,
        classnames=prompt_names, templates=templates, dataset=None,
        layer_index=m["layer_txt"], alignment_layer=at,
        num_classes_per_batch=8, device=device,
        pool_txt="none", save_path=None, token_level=True,
    ).to(device)
    text_feats = F.normalize(text_feats.float(), dim=-1)  # (C, K)
    C = text_feats.shape[0]

    print(f"\n  Text classifier shape: {text_feats.shape}", flush=True)
    print(f"  Text feat stats: mean={text_feats.mean():.4f}, std={text_feats.std():.4f}, "
          f"min={text_feats.min():.4f}, max={text_feats.max():.4f}", flush=True)

    for idx in sample_indices:
        pil_img, pil_mask = ds[idx]
        img_t = transform(pil_img).unsqueeze(0).float().to(device)
        gt = np.array(pil_mask)

        with torch.no_grad():
            lvm_out = vision_model(img_t)
            feats = list(lvm_out.values())[m["layer_img"]].squeeze(0)
            patches = feats[1:, :]
            p_n = F.normalize(patches, dim=-1)
            sim = p_n @ anchors.T  # (P, K)

            # Raw path
            raw_feats = F.normalize(sim, dim=-1)  # L2 norm in run_eval
            raw_logits = raw_feats @ text_feats.T  # (P, C)

            # Softmax path
            soft_feats = F.softmax(sim / tau, dim=-1)  # softmax over anchors
            soft_feats_n = F.normalize(soft_feats, dim=-1)  # L2 norm in run_eval
            soft_logits = soft_feats_n @ text_feats.T  # (P, C)

        P = sim.shape[0]
        h = int(round(math.sqrt(P)))

        # Pick a specific patch (center)
        center = P // 2
        corner = 0
        edge = h  # first patch of second row

        print(f"\n  --- Image {idx} (P={P}, h={h}) ---", flush=True)

        # Overall similarity distribution
        print(f"  Raw sim (patch@anchor): mean={sim.mean():.4f}, std={sim.std():.4f}, "
              f"range=[{sim.min():.4f}, {sim.max():.4f}]", flush=True)

        # After softmax: distribution
        soft = F.softmax(sim / tau, dim=-1)
        print(f"  After softmax(dim=-1): mean={soft.mean():.6f}, std={soft.std():.6f}, "
              f"range=[{soft.min():.6f}, {soft.max():.6f}]", flush=True)
        print(f"    expected uniform = {1/K:.6f}", flush=True)
        print(f"    max/uniform ratio = {soft.max().item() / (1/K):.2f}x", flush=True)

        # Entropy per patch
        ent = -(soft * torch.log(soft + 1e-10)).sum(dim=-1)  # (P,)
        max_ent = math.log(K)
        print(f"    entropy: mean={ent.mean():.3f}/{max_ent:.3f} "
              f"({100*ent.mean()/max_ent:.1f}%), "
              f"min={ent.min():.3f} ({100*ent.min()/max_ent:.1f}%)", flush=True)

        # How many anchors have >2x uniform probability?
        above_2x = (soft > 2/K).sum(dim=-1).float()
        above_5x = (soft > 5/K).sum(dim=-1).float()
        print(f"    anchors >2x uniform: mean={above_2x.mean():.1f}/512, "
              f"anchors >5x uniform: mean={above_5x.mean():.1f}/512", flush=True)

        # Now compare the logits
        print(f"\n  Raw logits (patch→class):", flush=True)
        print(f"    mean={raw_logits.mean():.4f}, std={raw_logits.std():.4f}, "
              f"range=[{raw_logits.min():.4f}, {raw_logits.max():.4f}]", flush=True)

        print(f"  Softmax logits (patch→class):", flush=True)
        print(f"    mean={soft_logits.mean():.4f}, std={soft_logits.std():.4f}, "
              f"range=[{soft_logits.min():.4f}, {soft_logits.max():.4f}]", flush=True)

        # Class discrimination: for each patch, how much does top class beat second?
        raw_sorted = raw_logits.sort(dim=-1, descending=True).values
        raw_margin = (raw_sorted[:, 0] - raw_sorted[:, 1])
        soft_sorted = soft_logits.sort(dim=-1, descending=True).values
        soft_margin = (soft_sorted[:, 0] - soft_sorted[:, 1])
        print(f"\n  Top1-Top2 margin (class discrimination):", flush=True)
        print(f"    Raw:     mean={raw_margin.mean():.4f}, std={raw_margin.std():.4f}", flush=True)
        print(f"    Softmax: mean={soft_margin.mean():.4f}, std={soft_margin.std():.4f}", flush=True)
        print(f"    Ratio:   {soft_margin.mean()/raw_margin.mean():.2f}x", flush=True)

        # Do they agree on predictions?
        raw_preds = raw_logits.argmax(dim=-1)
        soft_preds = soft_logits.argmax(dim=-1)
        agree = (raw_preds == soft_preds).float().mean()
        print(f"\n  Prediction agreement: {100*agree:.1f}%", flush=True)

    del vision_model, language_model
    torch.cuda.empty_cache()

print("\nDONE", flush=True)
