"""Merge feature shards into numpy memmap files (stream-and-delete).

Creates memory-mapped .npy files that can be loaded without using RAM.
Deletes each shard after copying its data to the memmap.

The trainer loads these via np.memmap — the OS pages in data on demand,
so only the current batch (~4096 samples) is in RAM at any time.

Usage:
    PYTHONPATH=. python scripts/laion/04b_merge_shards.py
"""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


SHARD_DIR = Path("results/features/laion/shards")
OUTPUT_DIR = Path("results/features/laion")


def main():
    # Discover shards
    shard_ids = sorted(set(
        int(f.name.split("_")[1])
        for f in SHARD_DIR.glob("shard_*_img.pt")
    ))
    n_shards = len(shard_ids)
    print(f"Found {n_shards} shards")

    # Pass 1: scan shapes
    print("Pass 1: scanning shard shapes...")
    max_txt_len = 0
    shard_sizes = []
    T_img, D = None, None
    for sid in tqdm(shard_ids, desc="Scanning"):
        txt = torch.load(str(SHARD_DIR / f"shard_{sid:04d}_txt.pt"), map_location="cpu")
        shard_sizes.append(txt.shape[0])
        max_txt_len = max(max_txt_len, txt.shape[1])
        if T_img is None:
            img = torch.load(str(SHARD_DIR / f"shard_{sid:04d}_img.pt"), map_location="cpu")
            T_img = img.shape[1]
            D = img.shape[2]
            del img
        del txt

    total_n = sum(shard_sizes)
    print(f"  Total samples: {total_n:,}")
    print(f"  Image shape: ({total_n}, {T_img}, {D})")
    print(f"  Text max len: {max_txt_len}, D={D}")

    # Pass 2a: image features → memmap (stream-and-delete)
    img_mmap_path = OUTPUT_DIR / "laion_image_tokens.npy"
    if img_mmap_path.exists():
        print(f"\n{img_mmap_path} already exists, skipping image merge")
    else:
        print(f"\nPass 2a: merging image features → {img_mmap_path}")
        img_mmap = np.memmap(str(img_mmap_path), dtype=np.float16, mode="w+",
                             shape=(total_n, T_img, D))
        offset = 0
        for sid in tqdm(shard_ids, desc="Image"):
            path = SHARD_DIR / f"shard_{sid:04d}_img.pt"
            img = torch.load(str(path), map_location="cpu").numpy()
            n = img.shape[0]
            img_mmap[offset:offset + n] = img
            offset += n
            del img
            os.remove(path)
        img_mmap.flush()
        del img_mmap
        sz = img_mmap_path.stat().st_size / 1e9
        print(f"  Saved: {sz:.1f} GB")

    # Pass 2b: text features → memmap (stream-and-delete)
    txt_mmap_path = OUTPUT_DIR / "laion_text_tokens.npy"
    if txt_mmap_path.exists():
        print(f"\n{txt_mmap_path} already exists, skipping text merge")
    else:
        print(f"\nPass 2b: merging text features → {txt_mmap_path}")
        txt_mmap = np.memmap(str(txt_mmap_path), dtype=np.float16, mode="w+",
                             shape=(total_n, max_txt_len, D))
        offset = 0
        for sid in tqdm(shard_ids, desc="Text"):
            txt_path = SHARD_DIR / f"shard_{sid:04d}_txt.pt"
            txt = torch.load(str(txt_path), map_location="cpu")
            n = txt.shape[0]
            T = txt.shape[1]
            if T < max_txt_len:
                txt = F.pad(txt, (0, 0, 0, max_txt_len - T))
            txt_mmap[offset:offset + n] = txt.numpy()
            offset += n
            del txt
            os.remove(txt_path)
        txt_mmap.flush()
        del txt_mmap
        sz = txt_mmap_path.stat().st_size / 1e9
        print(f"  Saved: {sz:.1f} GB")

    # Pass 2c: text masks → memmap (stream-and-delete)
    mask_mmap_path = OUTPUT_DIR / "laion_text_mask.npy"
    if mask_mmap_path.exists():
        print(f"\n{mask_mmap_path} already exists, skipping mask merge")
    else:
        print(f"\nPass 2c: merging text masks → {mask_mmap_path}")
        mask_mmap = np.memmap(str(mask_mmap_path), dtype=np.int64, mode="w+",
                              shape=(total_n, max_txt_len))
        offset = 0
        for sid in tqdm(shard_ids, desc="Mask"):
            mask_path = SHARD_DIR / f"shard_{sid:04d}_mask.pt"
            mask = torch.load(str(mask_path), map_location="cpu")
            n = mask.shape[0]
            T = mask.shape[1]
            if T < max_txt_len:
                mask = F.pad(mask, (0, max_txt_len - T))
            mask_mmap[offset:offset + n] = mask.numpy()
            offset += n
            del mask
            os.remove(mask_path)
        mask_mmap.flush()
        del mask_mmap
        sz = mask_mmap_path.stat().st_size / 1e9
        print(f"  Saved: {sz:.1f} GB")

    # Save metadata
    meta = {"total_n": total_n, "T_img": T_img, "T_txt": max_txt_len, "D": D}
    torch.save(meta, str(OUTPUT_DIR / "laion_meta.pt"))
    print(f"\nMetadata: {meta}")

    # Clean up shard dir
    try:
        SHARD_DIR.rmdir()
        print("Cleaned up shard directory")
    except OSError:
        remaining = list(SHARD_DIR.iterdir())
        print(f"Shard directory has {len(remaining)} remaining files")

    print("\nDone!")
    df_out = os.popen("df -h / | tail -1").read().strip()
    print(f"Disk: {df_out}")


if __name__ == "__main__":
    main()
