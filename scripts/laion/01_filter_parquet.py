"""Filter LAION-18M parquet to high-quality subset using CLIP similarity threshold.

Outputs a filtered parquet with ~847K rows (similarity >= 0.39).

Usage:
    python scripts/laion/01_filter_parquet.py [--threshold 0.39]
"""
import argparse
import pandas as pd
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/laion/laion_subsampled_18m_no_nsfw.snappy.parquet")
    p.add_argument("--output", default="data/laion/laion_filtered.parquet")
    p.add_argument("--threshold", type=float, default=0.39)
    args = p.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"  Total rows: {len(df):,}")

    # Filter by CLIP similarity
    df_filtered = df[df["similarity"] >= args.threshold].reset_index(drop=True)
    print(f"  After filter (>= {args.threshold}): {len(df_filtered):,}")

    # Add a sequential index for image naming
    df_filtered["laion_idx"] = range(len(df_filtered))

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_parquet(args.output, index=False)
    print(f"  Saved to {args.output}")

    # Stats
    print(f"\n  Similarity: mean={df_filtered['similarity'].mean():.4f}, "
          f"min={df_filtered['similarity'].min():.4f}, "
          f"max={df_filtered['similarity'].max():.4f}")
    print(f"  Avg dimensions: {df_filtered['HEIGHT'].mean():.0f} x {df_filtered['WIDTH'].mean():.0f}")


if __name__ == "__main__":
    main()
