# Experiment Scripts

Reusable bash scripts for running the STRUCTURE alignment experiment suite on one encoder combination at a time. Each encoder combo lives in its own subdirectory; you `cd` into it and run the numbered training scripts.

## Layout

```
scripts/
├── README.md                  ← this file
├── common.env                 ← settings shared across all encoder combos
├── vits_minilm/               ← ViT-S/14 DINOv2 (384d) + MiniLM-L6 (384d)
│   ├── config.env             ← encoder-specific vars (models, dims)
│   ├── 01_train_baselines.sh  ← Linear, Linear+STR, MLP, MLP+STR, CSA, CSA+STR
│   ├── 02_train_ba.sh         ← CLS BA K=128/256/512, Token BA K=128/256/512
│   └── 03_train_freezealign.sh ← FreezeAlign, FreezeAlign+STR
└── vitl_roberta/              ← ViT-L/14 DINOv2 (1024d) + RoBERTa-Large (1024d)
    ├── config.env
    ├── 01_train_baselines.sh
    ├── 02_train_ba.sh
    └── 03_train_freezealign.sh
```

## Layer selection + feature extraction are automatic

`train_alignment.py` handles layer selection and feature extraction internally. Every config in `configs/` sets `layer_selection.best_only: true`, so the **first run per encoder combo** will:

1. Extract all-layer CLS features (~10 min for ViT-S)
2. Run mutual kNN to pick `(layer_img, layer_txt)`
3. Extract per-layer token features for the selected pair (~30 min for ViT-S)
4. Begin training

Subsequent runs hit the cache and skip steps 1–3. No manual `01_layer_selection.sh` / `02_extract_features.sh` scripts are needed.

## Conventions

Every script:

1. Sources `../common.env` then its local `config.env`.
2. Takes the GPU id as the first positional argument (default `0`) and exports `CUDA_VISIBLE_DEVICES`.
3. `cd`s to the project root and writes a timestamped log to `logs/<encoder_tag>_<stage>_<YYYYMMDD_HHMM>.log`.
4. Runs experiments sequentially and prints a completion summary with wall time.
5. Is idempotent — re-running skips anything already cached.

Training entry point is always `PYTHONPATH=. python src/train_alignment.py --config_path <yaml>`. CSA uses the same entry point (the trainer dispatches on `training.cca: true`). Token-level runs are selected by `training.token_level: true` in their config.

## Usage

```bash
# Run experiments (split across GPUs):
bash scripts/vits_minilm/01_train_baselines.sh 0 &     # GPU 0: baselines
bash scripts/vits_minilm/02_train_ba.sh 1 &            # GPU 1: BA experiments

# After GPU 0 finishes baselines:
bash scripts/vits_minilm/03_train_freezealign.sh 0 &   # GPU 0: FreezeAlign
```

The first run on each GPU triggers layer selection + extraction automatically.

## Prerequisites

- Conda env `structure` activated (Python 3.10, torch 2.1.2+cu118).
- `data/COCO/` populated with the real COCO 2014 split (train2014, val2014, annotations).
- Feature caches live under `results/features/`; checkpoints under `results/checkpoints/`; logs under `logs/`.

## Adding a new encoder combo

1. `cp -r scripts/vits_minilm scripts/<new_tag>`
2. Edit `scripts/<new_tag>/config.env`: set `VISION_MODEL`, `LANGUAGE_MODEL`, `DIM_IMG`, `DIM_TXT`, `ENCODER_TAG`, and alignment dims.
3. Create matching configs under `configs/<method>/<new_tag>/` (or update the `CONFIGS` arrays in the new scripts).
4. Run the three training scripts.
