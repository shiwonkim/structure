"""Select top K successfully downloaded LAION images by CLIP similarity score.

Produces a final parquet with only the rows we'll use for training.

Usage:
    python scripts/laion/03_select_top_k.py [--k 917000]
"""
import argparse
import csv
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="data/laion/laion_filtered_extended.parquet")
    p.add_argument("--status-csv", default="data/laion/download_status.csv")
    p.add_argument("--image-dir", default="data/laion/images")
    p.add_argument("--k", type=int, default=917000)
    p.add_argument("--output", default="data/laion/laion_final.parquet")
    args = p.parse_args()

    print(f"Loading parquet {args.parquet}...")
    df = pd.read_parquet(args.parquet)
    print(f"  Total rows: {len(df):,}")

    # Load download status
    print(f"Loading download status {args.status_csv}...")
    ok_indices = set()
    with open(args.status_csv) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[1] in ("ok", "exists"):
                ok_indices.add(int(row[0]))
    print(f"  Successfully downloaded: {len(ok_indices):,}")

    # Filter to only downloaded images
    df_ok = df[df["laion_idx"].isin(ok_indices)].copy()
    print(f"  Matched in parquet: {len(df_ok):,}")

    # Verify images actually exist on disk
    existing = set()
    for idx in df_ok["laion_idx"]:
        if Path(f"{args.image_dir}/{idx}.jpg").exists():
            existing.add(idx)
    df_ok = df_ok[df_ok["laion_idx"].isin(existing)]
    print(f"  Verified on disk: {len(df_ok):,}")

    # Select top K by CLIP similarity
    df_ok = df_ok.sort_values("similarity", ascending=False)
    if len(df_ok) < args.k:
        print(f"  WARNING: Only {len(df_ok):,} available, less than target {args.k:,}")
        df_final = df_ok
    else:
        df_final = df_ok.head(args.k)
    df_final = df_final.reset_index(drop=True)

    print(f"\n  Final selection: {len(df_final):,}")
    print(f"  Similarity range: {df_final['similarity'].min():.4f} - {df_final['similarity'].max():.4f}")
    print(f"  Mean similarity: {df_final['similarity'].mean():.4f}")

    df_final.to_parquet(args.output, index=False)
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
