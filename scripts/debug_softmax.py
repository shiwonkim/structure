"""Debug softmax inconsistency between ViT-S and ViT-B seg eval."""

import torch, torch.nn.functional as F, os, math, timm
import numpy as np
from timm.data import resolve_data_config
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms, datasets
from src.evaluation.zero_shot_segmentation import load_config

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0")

models = {
    "vits": {
        "config": "configs/ba/vits_minilm/token_k512.yaml",
        "ckpt": "results/alignment-sentence_transformers_all_MiniLM_L6_v2-"
                "vit_small_patch14_dinov2.lvd142m-wobbly-water-15/"
                "(11, 6)_0.2739/checkpoints/checkpoint-epoch490.pth",
        "layer_img": 11,
    },
    "vitb": {
        "config": "configs/ba/vitb_mpnet/token_k512.yaml",
        "ckpt": "results/alignment-sentence_transformers_all_mpnet_base_v2-"
                "vit_base_patch14_dinov2.lvd142m-rich-flower-83/"
                "(11, 12)_0.2852/checkpoints/checkpoint-epoch408.pth",
        "layer_img": 11,
    },
}

print("=" * 70, flush=True)
print("1. TRAINING CONFIGURATION", flush=True)
print("=" * 70, flush=True)

for name, m in models.items():
    cfg = load_config(m["config"])
    kwargs = cfg["training"].get("alignment_layer_kwargs", {})
    tau = kwargs.get("pool_temperature", 0.05)
    K = kwargs.get("num_anchors", kwargs.get("dim_alignment", 256))
    print(f"  {name}: pool_temperature={tau}, K={K}", flush=True)

    ckpt = torch.load(m["ckpt"], map_location="cpu", weights_only=False)
    ai = ckpt["alignment_image"]
    # Check if tau is stored on checkpoint
    if hasattr(ai, "pool_temperature"):
        print(f"    checkpoint pool_temperature = {ai.pool_temperature}", flush=True)
    if hasattr(ai, "pool_method"):
        print(f"    checkpoint pool_method = {ai.pool_method}", flush=True)
    # Check anchor shape
    print(f"    anchors shape = {ai.anchors.shape}", flush=True)
    print(f"    anchors dtype = {ai.anchors.dtype}", flush=True)
    m["alignment_image"] = ai
    m["cfg"] = cfg

print(flush=True)
print("=" * 70, flush=True)
print("2. FEATURE SCALE ANALYSIS (10 random VOC images)", flush=True)
print("=" * 70, flush=True)

# Load VOC
from src.evaluation.zero_shot_segmentation import build_dataset
ds, ds_spec = build_dataset("voc2012", data_root="data/pascal_voc", download=False)

for name, m in models.items():
    cfg = m["cfg"]
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

    ai = m["alignment_image"].to(device).eval()
    anchors = F.normalize(ai.anchors, dim=-1).detach()
    tau = ai.pool_temperature if hasattr(ai, "pool_temperature") else 0.05

    print(f"\n  --- {name} (D={anchors.shape[1]}, K={anchors.shape[0]}, τ={tau}) ---", flush=True)

    all_sims = []
    all_entropies_softmax_anchors = []  # dim=-1 (over anchors)
    all_entropies_softmax_tokens = []   # dim=1 (over tokens, training CAP dim)

    np.random.seed(42)
    indices = np.random.choice(len(ds), 10, replace=False)

    for idx in indices:
        pil_img, _ = ds[idx]
        img_t = transform(pil_img).unsqueeze(0).float().to(device)
        with torch.no_grad():
            lvm_out = vision_model(img_t)
            feats = list(lvm_out.values())[m["layer_img"]].squeeze(0)  # (T, D)
            patches = feats[1:, :]  # (P, D)
            p_n = F.normalize(patches, dim=-1)
            sim = p_n @ anchors.T  # (P, K)
            all_sims.append(sim.cpu())

            # Entropy of softmax over anchors (dim=-1) — what anchor_codebook uses
            probs_a = F.softmax(sim / tau, dim=-1)
            ent_a = -(probs_a * torch.log(probs_a + 1e-10)).sum(dim=-1)  # (P,)
            all_entropies_softmax_anchors.append(ent_a.cpu())

            # Entropy of softmax over tokens (dim=0) — what training CAP uses
            probs_t = F.softmax(sim / tau, dim=0)
            ent_t = -(probs_t * torch.log(probs_t + 1e-10)).sum(dim=0)  # (K,)
            all_entropies_softmax_tokens.append(ent_t.cpu())

    sims = torch.cat(all_sims)  # (N*P, K)
    print(f"  Similarity stats (patch @ anchor):", flush=True)
    print(f"    min={sims.min():.4f}, max={sims.max():.4f}, "
          f"mean={sims.mean():.4f}, std={sims.std():.4f}", flush=True)
    print(f"    median={sims.median():.4f}, "
          f"p5={sims.quantile(0.05):.4f}, p95={sims.quantile(0.95):.4f}", flush=True)

    # Per-patch: max similarity to best anchor
    max_per_patch = sims.max(dim=-1).values
    print(f"  Per-patch max anchor similarity:", flush=True)
    print(f"    mean={max_per_patch.mean():.4f}, std={max_per_patch.std():.4f}", flush=True)

    ent_a = torch.cat(all_entropies_softmax_anchors)
    max_ent_a = math.log(anchors.shape[0])
    print(f"  Entropy of softmax(sim/τ, dim=anchors):", flush=True)
    print(f"    mean={ent_a.mean():.3f}, std={ent_a.std():.3f} "
          f"(max possible={max_ent_a:.3f}, ratio={ent_a.mean()/max_ent_a:.3f})", flush=True)

    ent_t = torch.cat(all_entropies_softmax_tokens)
    P = all_sims[0].shape[0]
    max_ent_t = math.log(P)
    print(f"  Entropy of softmax(sim/τ, dim=tokens):", flush=True)
    print(f"    mean={ent_t.mean():.3f}, std={ent_t.std():.3f} "
          f"(max possible={max_ent_t:.3f}, ratio={ent_t.mean()/max_ent_t:.3f})", flush=True)

    del vision_model
    torch.cuda.empty_cache()

print(flush=True)
print("=" * 70, flush=True)
print("4. SOFTMAX DIMENSION VERIFICATION", flush=True)
print("=" * 70, flush=True)

# Training CAP
import inspect
from src.alignment.bridge_anchor_token import BridgeAnchorTokenAlignmentLayer
src = inspect.getsource(BridgeAnchorTokenAlignmentLayer.forward)
# Find the softmax line
for line in src.split("\n"):
    if "softmax" in line and "dim" in line:
        print(f"  Training CAP: {line.strip()}", flush=True)

# Seg inference anchor_codebook
from src.evaluation.zero_shot_segmentation import BAAnchorCodebookMethod
src2 = inspect.getsource(BAAnchorCodebookMethod.get_patch_features)
for line in src2.split("\n"):
    if "softmax" in line and "dim" in line:
        print(f"  Seg anchor_codebook: {line.strip()}", flush=True)

print(flush=True)
print("=" * 70, flush=True)
print("5. LEARNABLE TAU CHECK", flush=True)
print("=" * 70, flush=True)

for name, m in models.items():
    ai = m["alignment_image"]
    tau_param = None
    for pname, param in ai.named_parameters():
        if "temperature" in pname or "tau" in pname:
            tau_param = (pname, param)
    if tau_param:
        print(f"  {name}: learnable tau found: {tau_param[0]} = {tau_param[1].item():.6f}", flush=True)
    else:
        print(f"  {name}: no learnable tau (fixed at pool_temperature={getattr(ai, 'pool_temperature', 'N/A')})", flush=True)

    # Also check all parameter names
    print(f"    All params: {[n for n, _ in ai.named_parameters()]}", flush=True)

print("\nDONE", flush=True)
