# LAION-addition: deferred work & reimplementation plan

> **Status (2026-06-19):** All LAION-addition code is **deferred on the `serverB`
> branch** and is intentionally **NOT in `main`**. LAION is in the paper
> appendix, so this code must be reimplemented and merged back. This doc is the
> handoff for that future work.

---

## 1. Why it's not in `main`

During the serverA/serverB → main integration we took `serverA`'s clean trainer
(`src/trainers/alignment_trainer.py`) and **dropped serverB's LAION/memmap
additions**, because serverB's memory approach was flawed: it still **OOM'd above
~80K LAION rows**, so it isn't a real scaling solution. Rather than carry broken
code into `main`, we deferred all of it for a clean reimplementation.

## 2. What is deferred (all on `origin/serverB`, absent from `main`)

| File | What it is |
|---|---|
| `scripts/laion/01_filter_parquet.py` … `04d_extract_cls_from_memmap.py` | LAION data pipeline: filter parquet → download images → select top-K → extract ViT-L/RoBERTa token features → memmap shards → CLS/avg extraction. **Trainer-independent; reusable as-is.** |
| `scripts/laion/05_train_data_scaling.py` | The **actual** LAION training driver used for the appendix (COCO + LAION sweep 80K…917K). Calls `trainer.fit(n_random_additional_feats=N)` with the memmap config. Coupled to the dropped trainer path. |
| `configs/ba/vitl_roberta/token_k512_laion.yaml` | Config with `add_img_feat_paths` / `add_txt_feat_paths` / `add_meta_path` (`.npy` memmap) that triggers the trainer's LAION-loading branch. |
| `src/utils/memmap_features.py` (`ConcatFeatureStore`) | Disk-backed lazy concat of in-RAM COCO + on-disk LAION memmap. **This part is correct** (see §4). |
| `alignment_trainer.py` LAION additions | `_torch_load_shared_mmap` (MAP_SHARED loader), the `~@1707` LAION-loading block, and the `_is_lazy` per-batch index branches. |
| `src/train_laion_addition_alignment.py` | **Unused** initial-commit template — no script ever referenced it. Not the code that produced any results. (Real runs used `05_train_data_scaling.py`.) |

### Retrieve the deferred code later
```bash
git fetch origin
git show origin/serverB:scripts/laion/05_train_data_scaling.py        # view one file
git checkout origin/serverB -- scripts/laion/ \
    configs/ba/vitl_roberta/token_k512_laion.yaml \
    src/utils/memmap_features.py                                       # pull into tree
git diff origin/main origin/serverB -- src/trainers/alignment_trainer.py  # see the trainer LAION diff
```

## 3. Root cause of the OOM (>80K LAION)

The trainer picks an **in-RAM path** for moderate sizes and only uses the
disk-backed store when huge:

```python
est_bytes = add_n_use * (T_img + T_txt) * D * 2     # counts LAION only
est_gb = est_bytes / 1e9
RAM_THRESHOLD_GB = 170
if est_gb < RAM_THRESHOLD_GB:        # ~80K LAION ≈ 52GB → this branch
    l_add_img = torch.from_numpy(np.array(img_mmap[idx]))   # materialize LAION in RAM
    layer_image_features_train = torch.cat((COCO, l_add_img), dim=0)  # new COCO+LAION tensor
```

Three problems, all in the **trainer's usage** (not in `ConcatFeatureStore`):

1. **`torch.cat(COCO, LAION)`** allocates a brand-new tensor while both sources
   still exist → **peak ≈ 2×(COCO+LAION)**. ViT-L COCO text tokens alone are
   ~156 GB → instant OOM on a ~136 GB machine.
2. **`RAM_THRESHOLD_GB = 170` is mis-calibrated** — it estimates LAION size only,
   ignoring the 156 GB COCO already in RAM and the cat's 2× peak. So it picks the
   RAM path even when it can't fit. (Likely tuned on small encoders where the
   COCO cache is ~20–39 GB, not ViT-L's 156 GB.)
3. **`F.pad` on the whole COCO text tensor** (to match LAION's seq length)
   materializes another full copy of the 156 GB tensor — and runs even on the
   disk path.

## 4. What's already correct

`src/utils/memmap_features.py` (`ConcatFeatureStore`) is the **right building
block**: it reads only the accessed rows from the memmap per batch (int / slice /
fancy-index), with a contiguous fast-path and `T_max` padding. It is genuinely
lazy and low-RAM. The bug is that the trainer **only uses it above 170 GB** while
defaulting to the in-RAM `torch.cat` path. Route everything through it and the
OOM goes away. (Minor robustness gaps in the class — ignores slice `step`,
no negative indices, scattered reads are slow on HDD — none matter for the
training use case.)

## 5. Reimplementation plan (low RAM **and** no speed regression)

The previous code was two extremes: all-RAM (fast, OOMs) vs naive random-access
memmap (no OOM, but **far slower** — that's why the threshold existed). The right
answer is in between:

1. **Virtual concat always** — never `torch.cat` COCO+LAION into one tensor.
   Keep them separate (à la `ConcatFeatureStore`) and index per batch. Kills the
   2× peak. Drop the 170 GB threshold entirely.
2. **Per-batch padding** — pad to common seq length inside the batch read, never
   `F.pad` the whole 156 GB COCO tensor.
3. **MAP_SHARED mmap + OS page cache** — load caches memory-mapped so committed
   RAM stays low while free RAM is used as page cache (RAM-speed reads after
   warmup). Server B's problem was `CommitLimit`, not physical RAM, so MAP_SHARED
   is what makes the page cache usable.
4. **Buffer / chunk shuffle** — instead of fully random per-batch reads (random
   disk seeks = the slowness), read large contiguous chunks sequentially and
   shuffle within a buffer. Converts random I/O → sequential I/O while keeping
   enough randomness for SGD.
5. **Async prefetch** — use a proper `IterableDataset` + `DataLoader`
   (`num_workers`, `pin_memory`, `prefetch_factor`) so disk I/O overlaps GPU
   compute and the GPU never stalls.

**Honest limit:** if the working set exceeds physical RAM and access is fully
random, you cannot hit RAM-speed. Steps 4 (sequentialize) + 5 (prefetch) are what
make it close — that's the real low-RAM-and-fast recipe.

## 6. Bring-back checklist

- [ ] Reimplement the trainer LAION path per §5 (no `torch.cat`, per-batch pad,
      always-lazy, buffer shuffle, prefetch).
- [ ] Keep `ConcatFeatureStore` (harden the minor gaps if convenient).
- [ ] `git checkout origin/serverB -- scripts/laion/` (data pipeline reusable).
- [ ] Rework `05_train_data_scaling.py` + `token_k512_laion.yaml` against the new
      path; drop the unused `train_laion_addition_alignment.py`.
- [ ] Verify it trains COCO+LAION up to ~1M rows without OOM, at acceptable
      speed, then merge to `main` in one cohesive commit.
