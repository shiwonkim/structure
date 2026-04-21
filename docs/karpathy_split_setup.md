# Karpathy Split Setup for COCO Retrieval Evaluation

## Problem

Our COCO retrieval evaluation was using the **full COCO 2014 val set (40,504 images)** instead of the standard **Karpathy 5K test split**. This made our retrieval R@1 numbers ~50% lower than published results (e.g., BA K=512: 20.6% on 40K vs 45.9% on 5K). All published papers (STRUCTURE, FreezeAlign, SAIL, CLIP) report on the Karpathy 5K split.

Flickr30k was already correct — it uses the Karpathy 1K test split via `data/flickr30k/test.txt`.

## What Changed

### 1. Downloaded Karpathy split file

```bash
cd data/COCO
wget "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip" -O caption_datasets.zip
unzip -q caption_datasets.zip -d karpathy_splits
```

This creates `data/COCO/karpathy_splits/dataset_coco.json` containing:
- `train`: 82,783 images (matches our training set)
- `val`: 5,000 images
- `test`: 5,000 images (all from val2014)
- `restval`: 30,504 images (remaining val2014)

### 2. Extracted test image IDs

```python
import json
with open('data/COCO/karpathy_splits/dataset_coco.json') as f:
    d = json.load(f)
test_ids = [img['cocoid'] for img in d['images'] if img['split'] == 'test']
with open('data/COCO/karpathy_test_ids.json', 'w') as f:
    json.dump(test_ids, f)
```

This creates `data/COCO/karpathy_test_ids.json` — a JSON list of 5,000 COCO image IDs. All are from val2014 (no overlap with our train2014 training data).

### 3. Added `coco_karpathy` dataset option

**File: `src/dataset_preparation/data_utils.py`**

The `get_datasets()` function now accepts `dataset="coco_karpathy"` in addition to `dataset="coco"`:

```python
elif dataset in ("coco", "coco_karpathy"):
    # ... same COCO loading as before ...

    if dataset == "coco_karpathy":
        karpathy_ids_path = coco_path / "karpathy_test_ids.json"
        if karpathy_ids_path.exists():
            import json as _json
            with open(karpathy_ids_path) as _f:
                test_ids = set(_json.load(_f))
            keep = val_dataset.df["image_path"].apply(
                lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]) in test_ids
            )
            val_dataset.df = val_dataset.df[keep].reset_index(drop=True)
            val_dataset.name = "coco_karpathy"
```

Key details:
- Loads the same `captions_val2014.json` as `coco`, then filters to 5K test images
- Sets `val_dataset.name = "coco_karpathy"` so feature caches are saved separately from the full 40K caches (cache path uses `.name`)
- Training dataset is unchanged — always the full 82K train set
- The image ID is extracted from the filename: `COCO_val2014_000000391895.jpg` → `391895`

### 4. No changes to training pipeline

The `features.dataset: "coco"` in configs controls the training data and is NOT affected. Only the retrieval evaluation dataset changes:

```yaml
# In configs:
evaluation:
    retrieval_datasets:
        - "coco_karpathy"  # standard 5K for paper
        - "flickr30"       # already Karpathy 1K
```

Or via `rerun_eval.py`:
```bash
python rerun_eval.py --rt coco_karpathy,flickr30 ...
```

### 5. Both splits coexist

- `--rt coco` → full 40K val, caches as `...-CocoCaptionDataset-eval-...npy`
- `--rt coco_karpathy` → Karpathy 5K test, caches as `...-coco_karpathy-eval-...npy`

No conflict. Same checkpoints work with either.

## Files on disk needed

```
data/COCO/
├── karpathy_splits/
│   └── dataset_coco.json          ← from caption_datasets.zip
├── karpathy_test_ids.json         ← extracted 5K test IDs
├── annotations/
│   ├── captions_train2014.json    ← existing
│   └── captions_val2014.json      ← existing
├── train2014/                     ← existing images
└── val2014/                       ← existing images (5K test subset lives here)
```

## Files modified in code

1. **`src/dataset_preparation/data_utils.py`** — added `coco_karpathy` branch in `get_datasets()`

## Verification

```python
from src.dataset_preparation.data_utils import get_datasets

_, full = get_datasets('coco', transform=None, root_dir='data/')
print(f'coco full val: {len(full)} samples, {full.df["image_path"].nunique()} unique images')
# → 202,654 samples, 40,504 unique images

_, karp = get_datasets('coco_karpathy', transform=None, root_dir='data/')
print(f'coco_karpathy: {len(karp)} samples, {karp.df["image_path"].nunique()} unique images')
# → 25,010 samples, 5,000 unique images
```

## Impact on results

Example: BA K=512 (ViT-S + MiniLM):

| Split | COCO I2T R@1 | COCO T2I R@1 |
|---|---|---|
| Full 40K val | 20.6% | 13.7% |
| Karpathy 5K test | **45.9%** | **32.9%** |

The 5K numbers are directly comparable to published results in STRUCTURE, FreezeAlign, SAIL papers.

## Backward compatibility

- Old `--rt coco` still works exactly as before (full 40K)
- Old feature caches are untouched
- Training is completely unaffected
- The `pool_method` backward compat fix (`getattr(self, "pool_method", "cap")` in `bridge_anchor_token.py`) ensures old BA checkpoints load correctly with the updated code
