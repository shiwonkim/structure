"""Download images for the filtered LAION subset.

Uses multiprocessing + requests with timeouts to handle dead links gracefully.
Saves images as JPEG to data/laion/images/{laion_idx}.jpg.
Tracks download status in a CSV for resumability.

Usage:
    python scripts/laion/02_download_images.py [--workers 32] [--timeout 10]
"""
import argparse
import csv
import io
import os
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import requests
from PIL import Image


def download_one(args):
    """Download a single image. Returns (laion_idx, status)."""
    laion_idx, url, out_path = args
    if os.path.exists(out_path):
        return (laion_idx, "exists")
    try:
        resp = requests.get(url, timeout=10, stream=True,
                           headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return (laion_idx, f"http_{resp.status_code}")
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        # Sanity check: skip tiny images
        if img.width < 10 or img.height < 10:
            return (laion_idx, "too_small")
        img.save(out_path, "JPEG", quality=95)
        return (laion_idx, "ok")
    except requests.exceptions.Timeout:
        return (laion_idx, "timeout")
    except Exception as e:
        return (laion_idx, f"error:{type(e).__name__}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/laion/laion_filtered.parquet")
    p.add_argument("--output-dir", default="data/laion/images")
    p.add_argument("--status-csv", default="data/laion/download_status.csv")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--timeout", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1000,
                   help="Save status CSV every N downloads")
    args = p.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} rows from {args.input}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load existing status for resumability
    done = set()
    status_path = Path(args.status_csv)
    if status_path.exists():
        with open(status_path) as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                done.add(int(row[0]))
        print(f"  Resuming: {len(done):,} already processed")

    # Build task list
    tasks = []
    for _, row in df.iterrows():
        idx = int(row["laion_idx"])
        if idx in done:
            continue
        out_path = str(out_dir / f"{idx}.jpg")
        tasks.append((idx, row["URL"], out_path))

    print(f"  {len(tasks):,} images to download ({len(done):,} already done)")

    if not tasks:
        print("Nothing to download.")
        return

    # Download with multiprocessing
    results = []
    counts = {"ok": 0, "exists": 0, "failed": 0}

    with open(status_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not done:
            writer.writerow(["laion_idx", "status"])

        with Pool(args.workers) as pool:
            for i, (idx, status) in enumerate(pool.imap_unordered(download_one, tasks)):
                writer.writerow([idx, status])
                if status in ("ok", "exists"):
                    counts["ok"] += 1
                else:
                    counts["failed"] += 1

                if (i + 1) % args.batch_size == 0:
                    f.flush()
                    total = counts["ok"] + counts["failed"]
                    print(f"  Progress: {total:,}/{len(tasks):,} | "
                          f"ok={counts['ok']:,} failed={counts['failed']:,}")

    total = counts["ok"] + counts["failed"]
    print(f"\nDone: {total:,} processed | ok={counts['ok']:,} failed={counts['failed']:,}")


if __name__ == "__main__":
    main()
