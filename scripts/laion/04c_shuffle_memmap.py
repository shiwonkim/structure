"""Pre-shuffle memmap by sequentially reading and writing in chunk order.

Since we can't fit a full copy on disk, we do a two-phase approach:
1. Split the memmap into small chunk files (sequential read — fast)
2. Delete original memmap
3. Write chunks back in shuffled order (sequential write — fast)

Each chunk is ~50 MB (1000 rows of image tokens), easily fits in RAM.

Usage:
    PYTHONPATH=. python scripts/laion/04c_shuffle_memmap.py
"""
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


INPUT_DIR = Path("results/features/laion")
CHUNK_DIR = INPUT_DIR / "shuffle_chunks"
CHUNK_SIZE = 1000  # rows per chunk


def split_to_chunks(mmap_path, shape, dtype, prefix):
    """Split memmap into chunk files via sequential read."""
    CHUNK_DIR.mkdir(exist_ok=True)
    src = np.memmap(str(mmap_path), dtype=dtype, mode="r", shape=shape)
    N = shape[0]
    n_chunks = (N + CHUNK_SIZE - 1) // CHUNK_SIZE

    for i in tqdm(range(n_chunks), desc=f"Split {prefix}"):
        start = i * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, N)
        chunk = np.array(src[start:end])
        np.save(str(CHUNK_DIR / f"{prefix}_{i:04d}.npy"), chunk)

    del src
    return n_chunks


def reassemble_shuffled(mmap_path, shape, dtype, prefix, chunk_perm, n_chunks,
                         within_perm_per_chunk):
    """Write chunks back in shuffled order (sequential write)."""
    dst = np.memmap(str(mmap_path), dtype=dtype, mode="w+", shape=shape)
    N = shape[0]
    offset = 0

    for i in tqdm(range(n_chunks), desc=f"Reassemble {prefix}"):
        src_chunk_id = chunk_perm[i]
        chunk = np.load(str(CHUNK_DIR / f"{prefix}_{src_chunk_id:04d}.npy"))
        # Shuffle within chunk
        if src_chunk_id in within_perm_per_chunk:
            chunk = chunk[within_perm_per_chunk[src_chunk_id]]
        n = chunk.shape[0]
        dst[offset:offset + n] = chunk
        offset += n
        # Delete chunk file immediately
        os.remove(str(CHUNK_DIR / f"{prefix}_{src_chunk_id:04d}.npy"))

    dst.flush()
    del dst


def main():
    meta = torch.load(str(INPUT_DIR / "laion_meta.pt"), weights_only=False)
    N = meta["total_n"]
    T_img = meta["T_img"]
    T_txt = meta["T_txt"]
    D = meta["D"]
    n_chunks = (N + CHUNK_SIZE - 1) // CHUNK_SIZE

    print(f"Samples: {N:,}, chunks: {n_chunks}, chunk_size: {CHUNK_SIZE}")

    # Generate shared permutations
    print("Generating permutations...")
    chunk_perm = np.random.permutation(n_chunks)
    within_perm = {}
    for i in range(n_chunks):
        start = i * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, N)
        size = end - start
        within_perm[i] = np.random.permutation(size)

    # Process each modality: split → delete original → reassemble shuffled
    files = [
        ("img", "laion_image_tokens.npy", (N, T_img, D), np.float16),
        ("txt", "laion_text_tokens.npy", (N, T_txt, D), np.float16),
        ("mask", "laion_text_mask.npy", (N, T_txt), np.int64),
    ]

    for prefix, fname, shape, dtype in files:
        mmap_path = INPUT_DIR / fname
        print(f"\n=== {fname} ({mmap_path.stat().st_size / 1e9:.1f} GB) ===")

        print("  Phase 1: splitting into chunks...")
        split_to_chunks(mmap_path, shape, dtype, prefix)

        print("  Phase 2: deleting original...")
        os.remove(str(mmap_path))

        print("  Phase 3: reassembling shuffled...")
        reassemble_shuffled(mmap_path, shape, dtype, prefix, chunk_perm,
                           n_chunks, within_perm)

        print(f"  Done: {mmap_path}")
        df_out = os.popen("df -h / | tail -1").read().strip().split()
        print(f"  Free space: {df_out[3]}")

    # Cleanup
    try:
        CHUNK_DIR.rmdir()
    except OSError:
        pass

    print("\nAll files shuffled!")


if __name__ == "__main__":
    main()
