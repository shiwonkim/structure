"""Extract CLS image features and masked-mean text features from the
existing LAION token memmap files. Output is small enough to fit in RAM.

Image: tokens[:, 0, :] → (N, D) CLS vector
Text: masked mean of tokens → (N, D) average vector

Usage:
    PYTHONPATH=. python scripts/laion/04d_extract_cls_from_memmap.py
"""
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

INPUT_DIR = Path("results/features/laion")
CHUNK = 10000


def main():
    meta = torch.load(str(INPUT_DIR / "laion_meta.pt"), weights_only=False)
    N = meta["total_n"]
    T_img = meta["T_img"]
    T_txt = meta["T_txt"]
    D = meta["D"]
    print(f"Samples: {N:,}, T_img={T_img}, T_txt={T_txt}, D={D}")

    # Image CLS: take position 0 from token sequence
    print("\nExtracting image CLS features...")
    img_mmap = np.memmap(str(INPUT_DIR / "laion_image_tokens.npy"),
                         dtype=np.float16, mode="r", shape=(N, T_img, D))
    img_cls = torch.empty(N, D, dtype=torch.float16)
    for start in trange(0, N, CHUNK, desc="Image CLS"):
        end = min(start + CHUNK, N)
        img_cls[start:end] = torch.from_numpy(np.array(img_mmap[start:end, 0, :]))
    del img_mmap

    out_path = INPUT_DIR / "laion_image_cls.pt"
    torch.save({"image_feats": img_cls}, str(out_path))
    print(f"  Saved: {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")
    del img_cls

    # Text masked mean
    print("\nExtracting text masked-mean features...")
    txt_mmap = np.memmap(str(INPUT_DIR / "laion_text_tokens.npy"),
                         dtype=np.float16, mode="r", shape=(N, T_txt, D))
    mask_mmap = np.memmap(str(INPUT_DIR / "laion_text_mask.npy"),
                          dtype=np.int64, mode="r", shape=(N, T_txt))
    txt_avg = torch.empty(N, D, dtype=torch.float16)
    for start in trange(0, N, CHUNK, desc="Text mean"):
        end = min(start + CHUNK, N)
        tokens = torch.from_numpy(np.array(txt_mmap[start:end])).float()
        masks = torch.from_numpy(np.array(mask_mmap[start:end])).float()
        # Masked mean: sum(tokens * mask) / sum(mask)
        masked = tokens * masks.unsqueeze(-1)
        lengths = masks.sum(dim=1, keepdim=True).clamp(min=1)
        avg = (masked.sum(dim=1) / lengths.squeeze(-1).unsqueeze(-1))
        txt_avg[start:end] = avg.half()
    del txt_mmap, mask_mmap

    out_path = INPUT_DIR / "laion_text_cls.pt"
    torch.save({"text_feats": txt_avg}, str(out_path))
    print(f"  Saved: {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")

    print("\nDone!")
    print(f"Total CLS features: {2 * N * D * 2 / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
