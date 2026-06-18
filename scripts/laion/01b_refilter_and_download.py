"""Re-filter LAION at a lower threshold to get more download candidates.

Preserves existing laion_idx for already-downloaded images by keeping
the old filtered set and appending new rows with fresh indices.

After all downloads complete, run 03_select_top_k.py to pick the best
847K by CLIP score from successfully downloaded images.

Usage:
    python scripts/laion/01b_refilter_and_download.py
"""
import pandas as pd
from pathlib import Path


def main():
    threshold = 0.37
    source_path = "data/laion/laion_subsampled_18m_no_nsfw.snappy.parquet"
    old_filtered_path = "data/laion/laion_filtered.parquet"
    new_filtered_path = "data/laion/laion_filtered_extended.parquet"

    print(f"Loading source {source_path}...")
    df_all = pd.read_parquet(source_path)

    print(f"Loading old filter {old_filtered_path}...")
    df_old = pd.read_parquet(old_filtered_path)
    old_sample_ids = set(df_old["SAMPLE_ID"].values)
    max_old_idx = df_old["laion_idx"].max()
    print(f"  Old filter: {len(df_old):,} rows, max laion_idx={max_old_idx}")

    # Get new candidates: >= threshold but NOT in old filter
    df_new_candidates = df_all[
        (df_all["similarity"] >= threshold) &
        (~df_all["SAMPLE_ID"].isin(old_sample_ids))
    ].copy()
    df_new_candidates = df_new_candidates.sort_values("similarity", ascending=False).reset_index(drop=True)
    # Assign new indices starting after old max
    df_new_candidates["laion_idx"] = range(max_old_idx + 1, max_old_idx + 1 + len(df_new_candidates))
    print(f"  New candidates (>= {threshold}, not in old): {len(df_new_candidates):,}")

    # Combine
    df_extended = pd.concat([df_old, df_new_candidates], ignore_index=True)
    df_extended.to_parquet(new_filtered_path, index=False)
    print(f"  Total extended: {len(df_extended):,}")
    print(f"  Saved to {new_filtered_path}")

    existing_images = len(list(Path("data/laion/images").glob("*.jpg")))
    print(f"\n  Existing downloaded images: {existing_images:,}")
    print(f"  New images to download: ~{len(df_new_candidates):,}")


if __name__ == "__main__":
    main()
