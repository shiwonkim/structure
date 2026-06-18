"""Extract token-level features from LAION images using ViT-L DINOv2 + RoBERTa.

Saves features incrementally in shards to avoid OOM. After extraction,
shards are concatenated into final feature files.

Resumable: skips already-completed shards.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/laion/04_extract_features.py \
        [--batch-size 64] [--gpu 0] [--shard-size 10000]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.text.models import load_llm, load_tokenizer
from timm import create_model
from timm.data import resolve_data_config
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.transforms as transforms


LVM_NAME = "vit_large_patch14_dinov2.lvd142m"
LLM_NAME = "sentence-transformers/all-roberta-large-v1"
IMG_SIZE = 224
LAYER_IMG = 23
LAYER_TXT = 24


class LaionShardDataset(Dataset):
    """Dataset for a shard (subset) of LAION images."""

    def __init__(self, df, image_dir, transform):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        laion_idx = int(row["laion_idx"])
        img_path = self.image_dir / f"{laion_idx}.jpg"
        try:
            img = Image.open(img_path).convert("RGB")
            img_t = self.transform(img)
        except Exception:
            img_t = torch.zeros(3, IMG_SIZE, IMG_SIZE)
        caption = str(row["TEXT"])
        return img_t, caption, laion_idx


def extract_shard(shard_df, image_dir, image_transform, vision_model,
                  language_model, tokenizer, device, batch_size, num_workers,
                  shard_dir, shard_id):
    """Extract features for one shard and save to disk immediately."""
    img_out = shard_dir / f"shard_{shard_id:04d}_img.pt"
    txt_out = shard_dir / f"shard_{shard_id:04d}_txt.pt"
    mask_out = shard_dir / f"shard_{shard_id:04d}_mask.pt"

    # Skip if already done
    if img_out.exists() and txt_out.exists() and mask_out.exists():
        return True

    dataset = LaionShardDataset(shard_df, image_dir, image_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                           num_workers=num_workers, pin_memory=True,
                           shuffle=False, drop_last=False)

    shard_img = []
    shard_txt = []
    shard_mask = []

    with torch.no_grad():
        for batch_imgs, batch_captions, batch_indices in dataloader:
            # Image features
            batch_imgs = batch_imgs.to(device)
            lvm_out = vision_model(batch_imgs)
            layer_key = list(lvm_out.keys())[LAYER_IMG]
            img_tokens = lvm_out[layer_key].half().cpu()
            shard_img.append(img_tokens)

            # Text features
            tokens = tokenizer(
                list(batch_captions),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            input_ids = tokens["input_ids"].to(device)
            attn_mask = tokens["attention_mask"].to(device)

            out = language_model(input_ids=input_ids, attention_mask=attn_mask)
            hidden = torch.stack(out["hidden_states"]).permute(1, 0, 2, 3)
            txt_tokens = hidden[:, LAYER_TXT, :, :].half().cpu()
            txt_mask = attn_mask.cpu()

            shard_txt.append(txt_tokens)
            shard_mask.append(txt_mask)

            del lvm_out, out, hidden
            torch.cuda.empty_cache()

    # Concatenate shard
    shard_img = torch.cat(shard_img, dim=0)

    # Pad text to uniform length within shard
    max_txt_len = max(t.shape[1] for t in shard_txt)
    padded_txt = []
    padded_mask = []
    for txt, mask in zip(shard_txt, shard_mask):
        B, T, D = txt.shape
        if T < max_txt_len:
            txt = F.pad(txt, (0, 0, 0, max_txt_len - T))
            mask = F.pad(mask, (0, max_txt_len - T))
        padded_txt.append(txt)
        padded_mask.append(mask)
    shard_txt = torch.cat(padded_txt, dim=0)
    shard_mask = torch.cat(padded_mask, dim=0)

    # Save shard
    torch.save(shard_img, str(img_out))
    torch.save(shard_txt, str(txt_out))
    torch.save(shard_mask, str(mask_out))

    return False


def merge_shards(shard_dir, output_dir, n_shards):
    """Concatenate all shards into final feature files."""
    img_out = output_dir / "laion_image_tokens.pt"
    txt_out = output_dir / "laion_text_tokens.pt"
    mask_out = output_dir / "laion_text_mask.pt"

    if img_out.exists() and txt_out.exists() and mask_out.exists():
        print("Final files already exist. Skipping merge.")
        return

    print("Merging shards...")
    all_img = []
    all_txt = []
    all_mask = []

    # Find max text length across all shards
    max_txt_len = 0
    for i in range(n_shards):
        txt = torch.load(str(shard_dir / f"shard_{i:04d}_txt.pt"), map_location="cpu")
        max_txt_len = max(max_txt_len, txt.shape[1])
    print(f"  Max text length across shards: {max_txt_len}")

    for i in tqdm(range(n_shards), desc="Merging"):
        img = torch.load(str(shard_dir / f"shard_{i:04d}_img.pt"), map_location="cpu")
        txt = torch.load(str(shard_dir / f"shard_{i:04d}_txt.pt"), map_location="cpu")
        mask = torch.load(str(shard_dir / f"shard_{i:04d}_mask.pt"), map_location="cpu")

        # Pad text to global max length
        if txt.shape[1] < max_txt_len:
            txt = F.pad(txt, (0, 0, 0, max_txt_len - txt.shape[1]))
            mask = F.pad(mask, (0, max_txt_len - mask.shape[1]))

        all_img.append(img)
        all_txt.append(txt)
        all_mask.append(mask)

    all_img = torch.cat(all_img, dim=0)
    all_txt = torch.cat(all_txt, dim=0)
    all_mask = torch.cat(all_mask, dim=0)

    print(f"  Image: {all_img.shape}")
    print(f"  Text: {all_txt.shape}")
    print(f"  Mask: {all_mask.shape}")

    torch.save({"image_feats": all_img}, str(img_out))
    torch.save({"text_feats": all_txt}, str(txt_out))
    torch.save({"mask": all_mask}, str(mask_out))

    print(f"  Saved to {output_dir}")
    for f in [img_out, txt_out, mask_out]:
        sz = f.stat().st_size / 1e9
        print(f"    {f.name}: {sz:.1f} GB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="data/laion/laion_final.parquet")
    p.add_argument("--image-dir", default="data/laion/images")
    p.add_argument("--output-dir", default="results/features/laion")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--shard-size", type=int, default=10000,
                   help="Samples per shard (saved to disk incrementally)")
    p.add_argument("--merge-only", action="store_true",
                   help="Skip extraction, just merge existing shards")
    p.add_argument("--shard-start", type=int, default=None,
                   help="First shard index (for multi-GPU splitting)")
    p.add_argument("--shard-end", type=int, default=None,
                   help="Last shard index exclusive (for multi-GPU splitting)")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    output_dir = Path(args.output_dir)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Load parquet
    df = pd.read_parquet(args.parquet)
    N = len(df)
    n_shards = (N + args.shard_size - 1) // args.shard_size
    print(f"Loaded {N:,} samples, {n_shards} shards of {args.shard_size}")

    if not args.merge_only:
        # Load models
        print("Loading vision model...")
        vision_model = create_model(LVM_NAME, pretrained=True, img_size=IMG_SIZE)
        data_config = resolve_data_config(vision_model.pretrained_cfg, model=vision_model)
        data_config["input_size"] = (3, IMG_SIZE, IMG_SIZE)
        data_config["crop_pct"] = 1.0
        return_nodes = [f"blocks.{i}.add_1" for i in range(len(vision_model.blocks))]
        vision_model = create_feature_extractor(vision_model, return_nodes=return_nodes)
        vision_model = vision_model.eval().to(device)

        image_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_config["mean"], std=data_config["std"]),
        ])

        print("Loading language model...")
        language_model = load_llm(LLM_NAME).to(device)
        tokenizer = load_tokenizer(LLM_NAME)

        # Extract shards (optionally a subset for multi-GPU)
        shard_start = args.shard_start if args.shard_start is not None else 0
        shard_end = args.shard_end if args.shard_end is not None else n_shards
        print(f"Processing shards {shard_start} to {shard_end - 1} (of {n_shards})")
        for shard_id in tqdm(range(shard_start, shard_end), desc="Shards"):
            start = shard_id * args.shard_size
            end = min(start + args.shard_size, N)
            shard_df = df.iloc[start:end]

            skipped = extract_shard(
                shard_df, args.image_dir, image_transform,
                vision_model, language_model, tokenizer,
                device, args.batch_size, args.num_workers,
                shard_dir, shard_id,
            )
            if skipped:
                continue

        # Free GPU
        del vision_model, language_model
        torch.cuda.empty_cache()

    # Merge shards
    merge_shards(shard_dir, output_dir, n_shards)
    print("\nDone!")


if __name__ == "__main__":
    main()
