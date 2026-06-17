"""Find image pairs that share anchor 'specialists' for the qualitative figure.

Goal: pick pairs (i, j) where the SAME small set of anchors (e.g. top-3) is
strongly activated in both images, while inside each image those anchors
attend to DIFFERENT spatial regions / different caption tokens. That way the
figure shows simultaneously:
    - within-image specialization (3 anchors → 3 distinct parts/words)
    - cross-image reuse        (same 3 anchor indices → semantically aligned roles)

Pipeline:
    1. Sample N COCO val images with rich captions (>=3 content words).
    2. For each image, compute (P, K) image-anchor attention and per-anchor
       activation a_img(k) = max_p attn[p, k].
    3. Score all pairs by:
         overlap        = |topK_i ∩ topK_j|
         specialization = pairwise (1 - cos) of the 3 shared anchors' patch
                          attention maps, averaged within each image
         caption_div    = 1 - jaccard(content words)
         profile_sim    = cos(a_i, a_j); fine-grained sweet spot ≈ 0.5–0.9
       score = overlap × min(spec_i, spec_j) × caption_div × gauss(profile_sim)
    4. Render a contact-sheet PNG of the top-K pairs, dump JSON metadata so
       you can feed picks straight into anchor_cross_modal.py --mode grid.

Usage:
    PYTHONPATH=. python scripts/viz/anchor_pair_finder.py \
        --config configs/ba/vitl_roberta/token_k512.yaml \
        --ckpt 'serverB/results/alignment-sentence_transformers_all_roberta_large_v1-vit_large_patch14_dinov2.lvd142m-polished-firefly-177/(23, 24)_nan/checkpoints/checkpoint-epoch373.pth' \
        --layer-img 23 --layer-txt 24 \
        --gpu 0 \
        --n-samples 250 --n-pairs 16 \
        --output drafts/figures/anchor_pair_candidates.png \
        --output-json drafts/figures/anchor_pair_candidates.json
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.alignment import *  # noqa: F401,F403
from src.core.src.utils.loader import Loader, merge_dicts
from src.models.text.models import load_llm, load_tokenizer
from timm import create_model
from timm.data import resolve_data_config
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.transforms as transforms

from scripts.viz.anchor_cross_modal import (
    _ensure_rgb,
    get_image_anchor_attention,
    get_text_anchor_attention,
)


STOPWORDS = set(
    "a an the of in on at to and or with for by from is are was were be been "
    "this that these those it its his her their there here some any all as "
    "into about over under near onto off out up down then than but if while "
    "very just also so such i you we they me my our your he she him them".split()
)


def content_words(caption):
    toks = [t.lower().strip(",.;:!?\"'()[]") for t in caption.split()]
    return [t for t in toks if t and t not in STOPWORDS and len(t) > 2]


def caption_ok(caption, min_words=7, min_content=3):
    words = caption.split()
    return len(words) >= min_words and len(content_words(caption)) >= min_content


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def cos_np(a, b, eps=1e-10):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def gaussian(x, mu, sigma):
    return float(np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2)))


def shared_top_anchors(a_i, a_j, top_k=10, n_shared=3):
    top_i = set(np.argsort(a_i)[-top_k:])
    top_j = set(np.argsort(a_j)[-top_k:])
    inter = top_i & top_j
    if len(inter) < n_shared:
        return None, len(inter)
    # rank intersection by combined activation
    inter_arr = np.array(sorted(inter))
    combined = a_i[inter_arr] + a_j[inter_arr]
    order = inter_arr[np.argsort(-combined)]
    return order[:n_shared].tolist(), len(inter)


def specialization_score(img_attn, anchors):
    """Mean pairwise (1 - cos) of patch attention maps for the given anchors.

    Higher = the anchors look at more distinct image regions.
    """
    profs = [img_attn[:, k] for k in anchors]
    pairs = [(i, j) for i in range(len(profs)) for j in range(i + 1, len(profs))]
    scores = [1.0 - cos_np(profs[i], profs[j]) for i, j in pairs]
    return float(np.mean(scores)) if scores else 0.0


def text_specialization_score(txt_attn, anchors):
    profs = [txt_attn[:, k] for k in anchors]
    pairs = [(i, j) for i in range(len(profs)) for j in range(i + 1, len(profs))]
    scores = [1.0 - cos_np(profs[i], profs[j]) for i, j in pairs]
    return float(np.mean(scores)) if scores else 0.0


def load_coco_val_pool(n_samples, seed):
    """Return list of dicts with keys: idx (in coco_data['images']), path, caption."""
    with open("data/COCO/annotations/captions_val2014.json") as f:
        coco = json.load(f)
    # build image_id -> first caption map
    cap_by_img = {}
    for a in coco["annotations"]:
        cap_by_img.setdefault(a["image_id"], []).append(a["caption"])
    pool = []
    for i, info in enumerate(coco["images"]):
        caps = cap_by_img.get(info["id"], [])
        if not caps:
            continue
        cap = caps[0].strip()
        if not caption_ok(cap):
            continue
        pool.append({
            "idx": i,
            "image_id": info["id"],
            "path": f"data/COCO/val2014/{info['file_name']}",
            "caption": cap,
        })
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n_samples]


def load_flickr_pool(n_samples, seed, split="test"):
    """Flickr30k pool. idx = position in data/flickr30k/<split>.txt
    (matches anchor_cross_modal.py --dataset flickr30 --image-idx convention).
    Caption = comment_number == "0" entry per image from results.csv.
    """
    import csv
    ids_path = Path(f"data/flickr30k/{split}.txt")
    with open(ids_path) as f:
        image_ids = [line.strip() for line in f if line.strip()]

    cap_by_img = {}
    with open("data/flickr30k/results.csv") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) < 3:
                continue
            img_file = row[0].strip()
            cnum = row[1].strip()
            cap = row[2].strip()
            if cnum == "0":
                cap_by_img[img_file.replace(".jpg", "")] = cap

    pool = []
    for i, img_id in enumerate(image_ids):
        cap = cap_by_img.get(img_id)
        if not cap:
            continue
        if not caption_ok(cap):
            continue
        pool.append({
            "idx": i,
            "image_id": img_id,
            "path": f"data/flickr30k/images/{img_id}.jpg",
            "caption": cap,
        })
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n_samples]


def load_dataset_pool(dataset, n_samples, seed):
    if dataset == "coco":
        return load_coco_val_pool(n_samples, seed)
    if dataset == "flickr30":
        return load_flickr_pool(n_samples, seed)
    raise ValueError(f"Unknown dataset: {dataset!r}")


def compute_attentions(samples, vision_model, image_transform, alignment_image,
                       layer_img, language_model, tokenizer, alignment_text,
                       layer_txt, device):
    """Add img_attn, txt_attn, token_strs, a_img to each sample dict."""
    t0 = time.time()
    for s in samples:
        s["pil"] = Image.open(s["path"]).convert("RGB")
        img_attn, _ = get_image_anchor_attention(
            s["pil"], vision_model, image_transform, alignment_image,
            layer_img, device,
        )
        txt_attn, _, token_strs, _ = get_text_anchor_attention(
            s["caption"], tokenizer, language_model, alignment_text,
            layer_txt, device,
        )
        s["img_attn"] = img_attn         # (P, K)
        s["txt_attn"] = txt_attn         # (T, K)
        s["token_strs"] = token_strs
        s["a_img"] = img_attn.max(axis=0)  # (K,)
    print(f"[attention] {len(samples)} samples in {time.time() - t0:.1f}s")


def score_pairs(samples, top_k=10, n_shared=3, profile_mu=0.7, profile_sigma=0.18):
    """Return list of dicts ranked best-first."""
    out = []
    n = len(samples)
    for i in range(n):
        si = samples[i]
        for j in range(i + 1, n):
            sj = samples[j]
            anchors, overlap = shared_top_anchors(
                si["a_img"], sj["a_img"], top_k=top_k, n_shared=n_shared,
            )
            if anchors is None:
                continue

            spec_i = specialization_score(si["img_attn"], anchors)
            spec_j = specialization_score(sj["img_attn"], anchors)
            tspec_i = text_specialization_score(si["txt_attn"], anchors)
            tspec_j = text_specialization_score(sj["txt_attn"], anchors)

            cw_i = content_words(si["caption"])
            cw_j = content_words(sj["caption"])
            cap_div = 1.0 - jaccard(cw_i, cw_j)

            prof_sim = cos_np(si["a_img"], sj["a_img"])
            prof_w = gaussian(prof_sim, profile_mu, profile_sigma)

            spec_min = min(spec_i, spec_j)
            tspec_min = min(tspec_i, tspec_j)

            score = (
                overlap
                * spec_min
                * (0.5 + 0.5 * tspec_min)  # token-level diversity (soft)
                * cap_div
                * prof_w
            )
            out.append({
                "i": i, "j": j,
                "anchors": anchors,
                "overlap": overlap,
                "spec_i": spec_i, "spec_j": spec_j,
                "tspec_i": tspec_i, "tspec_j": tspec_j,
                "cap_div": cap_div,
                "prof_sim": prof_sim,
                "score": score,
            })
    out.sort(key=lambda r: -r["score"])
    return out


def render_contact_sheet(samples, ranked_pairs, n_pairs, output_path,
                         thumb_size=240):
    """Each row: [img_i thumb | img_j thumb | text block]."""
    n_pairs = min(n_pairs, len(ranked_pairs))
    if n_pairs == 0:
        print("No pairs found.")
        return

    row_h = 2.4
    fig = plt.figure(figsize=(14, row_h * n_pairs))
    gs = gridspec.GridSpec(n_pairs, 3, width_ratios=[1, 1, 2.4],
                           hspace=0.5, wspace=0.05)

    for r, p in enumerate(ranked_pairs[:n_pairs]):
        si = samples[p["i"]]
        sj = samples[p["j"]]

        for col, s in enumerate([si, sj]):
            ax = fig.add_subplot(gs[r, col])
            img = np.array(s["pil"].resize((thumb_size, thumb_size)))
            ax.imshow(img)
            ax.set_title(f"#{s['idx']}", fontsize=9)
            ax.axis("off")

        ax_t = fig.add_subplot(gs[r, 2])
        ax_t.axis("off")
        ax_t.set_xlim(0, 1)
        ax_t.set_ylim(0, 1)
        anchors_str = ", ".join(str(a) for a in p["anchors"])
        header = (
            f"rank {r + 1}   score {p['score']:.3f}   "
            f"anchors [{anchors_str}]"
        )
        body = (
            f"overlap(top-10) = {p['overlap']}\n"
            f"img spec  i={p['spec_i']:.2f}  j={p['spec_j']:.2f}\n"
            f"txt spec  i={p['tspec_i']:.2f}  j={p['tspec_j']:.2f}\n"
            f"cap_div = {p['cap_div']:.2f}   profile_cos = {p['prof_sim']:.2f}\n\n"
            f"i (#{si['idx']}): {si['caption']}\n\n"
            f"j (#{sj['idx']}): {sj['caption']}"
        )
        ax_t.text(0.0, 0.97, header, transform=ax_t.transAxes,
                  fontsize=9, fontweight="bold", va="top", family="monospace")
        ax_t.text(0.0, 0.82, body, transform=ax_t.transAxes,
                  fontsize=8, va="top", family="monospace", wrap=True)

    plt.savefig(output_path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close()
    print(f"Saved contact sheet to {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--layer-img", type=int, default=23)
    p.add_argument("--layer-txt", type=int, default=24)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=200,
                   help="Number of COCO val images to consider.")
    p.add_argument("--n-pairs", type=int, default=16,
                   help="Number of top pairs to render.")
    p.add_argument("--top-k", type=int, default=10,
                   help="Image-level top-k anchors per image used to compute "
                        "the overlap set.")
    p.add_argument("--n-shared", type=int, default=3,
                   help="How many anchors per pair (target = 3 for the figure).")
    p.add_argument("--profile-mu", type=float, default=0.7,
                   help="Sweet-spot for cos(a_i, a_j); pairs near this similarity "
                        "are preferred (same category, different sample).")
    p.add_argument("--profile-sigma", type=float, default=0.18)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", choices=["coco", "flickr30"], default="flickr30",
                   help="Dataset pool to draw from (default: flickr30 test)")
    p.add_argument("--output", default="drafts/figures/anchor_pair_candidates.png")
    p.add_argument("--output-json",
                   default="drafts/figures/anchor_pair_candidates.json")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")

    with open(args.config) as f:
        cfg = Loader(f).get_single_data()
    cfg = merge_dicts(cfg.get("defaults", {}), cfg.get("overrides", {}))

    print(f"[ckpt] loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    alignment_image = ckpt["alignment_image"].eval().to(device)
    alignment_text = ckpt["alignment_text"].eval().to(device)

    lvm_name = cfg["alignment"]["lvm_model_name"]
    llm_name = cfg["alignment"]["llm_model_name"]
    img_size = int(cfg["features"].get("img_size", 224))
    print(f"[encoders] vision={lvm_name}  text={llm_name}  img_size={img_size}")

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

    print(f"[pool] sampling {args.n_samples} {args.dataset} images")
    samples = load_dataset_pool(args.dataset, args.n_samples, seed=args.seed)
    print(f"[pool] got {len(samples)} samples after caption filter")

    compute_attentions(
        samples, vision_model, image_transform, alignment_image, args.layer_img,
        language_model, tokenizer, alignment_text, args.layer_txt, device,
    )

    print(f"[score] scoring {len(samples) * (len(samples) - 1) // 2} pairs")
    ranked = score_pairs(
        samples, top_k=args.top_k, n_shared=args.n_shared,
        profile_mu=args.profile_mu, profile_sigma=args.profile_sigma,
    )
    print(f"[score] kept {len(ranked)} pairs with overlap >= {args.n_shared}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    render_contact_sheet(samples, ranked, args.n_pairs, args.output)

    # JSON metadata: only essentials so the file stays small
    meta = []
    for r, p in enumerate(ranked[: args.n_pairs]):
        si = samples[p["i"]]
        sj = samples[p["j"]]
        meta.append({
            "rank": r + 1,
            "score": p["score"],
            "image_idx_i": si["idx"],
            "image_idx_j": sj["idx"],
            "image_id_i": si["image_id"],
            "image_id_j": sj["image_id"],
            "path_i": si["path"],
            "path_j": sj["path"],
            "caption_i": si["caption"],
            "caption_j": sj["caption"],
            "anchors": p["anchors"],
            "overlap": p["overlap"],
            "spec_i": p["spec_i"], "spec_j": p["spec_j"],
            "tspec_i": p["tspec_i"], "tspec_j": p["tspec_j"],
            "cap_div": p["cap_div"], "profile_sim": p["prof_sim"],
        })
    with open(args.output_json, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {args.output_json}")

    if meta:
        anchors_str = ",".join(str(a) for a in meta[0]["anchors"])
        print("\nTop pair → ready-to-run grid command:")
        print(
            f"PYTHONPATH=. python scripts/viz/anchor_cross_modal.py \\\n"
            f"    --config {args.config} \\\n"
            f"    --ckpt '{args.ckpt}' \\\n"
            f"    --layer-img {args.layer_img} --layer-txt {args.layer_txt} \\\n"
            f"    --gpu {args.gpu} \\\n"
            f"    --mode grid \\\n"
            f"    --image-indices {meta[0]['image_idx_i']},{meta[0]['image_idx_j']} \\\n"
            f"    --anchor-overrides '{anchors_str};{anchors_str}' \\\n"
            f"    --output drafts/figures/picked_pair.png"
        )


if __name__ == "__main__":
    main()
