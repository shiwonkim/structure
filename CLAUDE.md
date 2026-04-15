# Bridge Anchors × STRUCTURE Integration

**Read this file first.** It is the entry point for every new Claude Code session on this repo. For deeper context, also scan `EXPERIMENTS.md` (batch ledger), `PROJECT_LOG.md` (chronological work log), and `IMPLEMENTATION.md` (design notes, file-level diffs).

## Project goal

Integrate **BridgeAnchors (BA)** — a cross-modal alignment method using K learnable anchor points with Cross-Attention Pooling (CAP) — into the **STRUCTURE** codebase, and produce a fair head-to-head comparison with existing alignment methods for a NeurIPS submission. Target table: 14 methods × 2 encoder scales × COCO 2014.

Upstream: https://github.com/mlbio-epfl/STRUCTURE (NeurIPS'25). Our fork adds BA (CLS + token variants), FreezeAlign as a token-level baseline, and a reorganized experiment harness.

## Current state (as of the most recent session)

- **Server A is running Batch 2** (small-model main table on COCO 2014, ViT-S + MiniLM). 10/14 runs finished:
  - Baselines batch (GPU 0): Linear, Linear+STR, MLP, MLP+STR, CSA, CSA+STR — all done.
  - BA batch (GPU 1): CLS BA K=128/256/512, Token BA K=128 — done. Token BA K=256 in progress; K=512 queued.
  - FreezeAlign batch (GPU 0 after baselines): `fa_d512` running; `fa_d512_struct` queued.
- **Headline result**: Token BA K=128 scores **16.30 / 11.30 / 47.73** on COCO I2T-R@1 / T2I-R@1 / I2T-R@10 — roughly 2× every CLS baseline (~8.5 / 6.1 / 32.6). Reproduces Batch-1 finding: CAP pooling is the differentiator, not the anchor head.
- **Server B is the next target**. Same codebase (rsync), fresh features, Batch 3 = the same 14-method table at ViT-L + RoBERTa-Large scale. Docker image `shiwonkim/structure-ba:v1` already pushed.

Check `EXPERIMENTS.md` and the latest `logs/vits_minilm_*_<timestamp>.log` files for live status.

## Environment

- Python 3.10, PyTorch 2.1.2+cu118, CUDA 11.8
- Conda env: `structure` (also baked into Docker image `shiwonkim/structure-ba:v1`)
- Key packages: timm 0.9.16, transformers 4.45.2, deepspeed 0.14.4, cca-zoo 2.5.0, wandb, loguru
- Server A: CUDA driver 470 → **requires** torch 2.1.2 (not 2.2.0) due to cuDNN 8.7 incompatibility
- Server B: CUDA driver 550, CUDA 12.4 — both torch versions work

## Directory structure

```
STRUCTURE/
├── CLAUDE.md                      ← you are here
├── EXPERIMENTS.md                 ← batch ledger (Batch 1 done, Batch 2 running, Batch 3 pending)
├── PROJECT_LOG.md                 ← chronological work log (update after every task)
├── IMPLEMENTATION.md              ← design notes, per-file diffs, rationale
├── Readme.md                      ← original STRUCTURE paper README
├── rerun_eval.py                  ← standalone post-hoc eval runner for trained checkpoints
├── configs/
│   ├── default.yaml               ← shared defaults, every config inherits via !include
│   ├── README.md                  ← configs layout + conventions
│   ├── _reference_structure/      ← original STRUCTURE configs (frozen, reference only)
│   ├── dryrun/                    ← smoke-test configs
│   ├── linear/<encoder>/          ← linear_d512{,_struct}
│   ├── mlp/<encoder>/             ← mlp_d512{,_struct}
│   ├── csa/<encoder>/             ← csa_d<sim>{,_struct}
│   ├── ba/<encoder>/              ← cls_k{128,256,512}, token_k{128,256,512}
│   └── freezealign/<encoder>/     ← fa_d512{,_struct}
├── scripts/
│   ├── README.md                  ← script conventions + how to add encoder combos
│   ├── common.env                 ← training defaults (epochs, BS, seed, STR hyperparams)
│   ├── vits_minilm/               ← ViT-S/14 DINOv2 + MiniLM-L6
│   │   ├── config.env             ← encoder-specific vars
│   │   ├── 01_train_baselines.sh  ← Linear, Linear+STR, MLP, MLP+STR, CSA, CSA+STR
│   │   ├── 02_train_ba.sh         ← CLS BA K=128/256/512, Token BA K=128/256/512
│   │   └── 03_train_freezealign.sh← FreezeAlign, FreezeAlign+STR
│   └── vitl_roberta/              ← ViT-L/14 + RoBERTa-Large (Batch 3, no runs yet)
│       └── (same layout)
├── src/
│   ├── train_alignment.py         ← THE entry point; dispatches trainer by config
│   ├── extract_features.py        ← standalone multi-model extraction (STRUCTURE paper method; we don't use it)
│   ├── measure_alignment.py       ← standalone mutual-kNN sweep over modelsets (we don't use it)
│   ├── alignment/
│   │   ├── alignment_factory.py   ← @AlignmentFactory.register() decorator
│   │   ├── base_alignment_layer.py← BaseAlignmentLayer(input_dim) → forward(z) → z'
│   │   ├── linear.py              ← LinearAlignmentLayer
│   │   ├── res_low_rank.py        ← ResLowRankHead (STRUCTURE's "MLP")
│   │   ├── bridge_anchor.py       ← BridgeAnchorAlignmentLayer (CLS, K×D anchors)
│   │   ├── bridge_anchor_token.py ← BridgeAnchorTokenAlignmentLayer (CAP + optional CLS attention prior)
│   │   └── freeze_align.py        ← FreezeAlignAlignmentLayer (token + CLS fallback via length-1 unsqueeze)
│   ├── trainers/
│   │   ├── alignment_trainer.py   ← main trainer: extraction, layer sel, fit, eval
│   │   └── csa_trainer.py         ← CSA (closed-form CCA) trainer, dispatched on training.cca=true
│   ├── evaluation/                ← zero-shot + retrieval (both CLS and token-level paths)
│   ├── dataset_preparation/       ← dataset loaders, incl. custom Flickr30k/COCO fixes
│   ├── models/                    ← timm vision, HF language loaders
│   ├── loss/                      ← CLIPLoss + STRUCTURE regularizer
│   └── core/                      ← vendored utils (don't modify)
├── data/
│   ├── COCO/                      ← real COCO 2014 (82,783 train + 40,504 val + annotations)
│   │   ├── train2014/   (13 GB)
│   │   ├── val2014/     (6.3 GB)
│   │   └── annotations/ (806 MB, real 2014 JSONs)
│   ├── flickr30k/                 ← Karpathy split (29,783 / 1K / 1K) + results.csv + images/
│   ├── imagenet/                  ← symlinked for zero-shot eval
│   └── (torchvision auto-downloads cifar10/100, stl10, dtd, flowers, gtsrb, country211, mnist, …)
├── results/
│   ├── features/                  ← cached feature tensors (CLS + token) per (model, dataset, split, layer)
│   ├── checkpoints/               ← alignment head weights per run
│   └── alignment-<llm>-<lvm>-<wandb-run-name>/  ← per-run output dirs
├── logs/                          ← timestamped training logs from scripts/
├── archive/
│   ├── 2026-04-14_coco2017_vits/  ← Batch 1 (wrong split) — logs, checkpoints, features, wandb
│   └── orig_structure_build/      ← obsolete STRUCTURE build system (Dockerfile, Makefile, common.mk, requirements.txt)
└── docker/                        ← Server B deployment (Dockerfile.structure, requirements.structure.txt)
```

## How to run experiments

**One command per encoder combo, one script per method family, GPU id as the only arg.** The scripts encapsulate the full 6 baselines / 6 BA / 2 FreezeAlign overnight queue.

```bash
# Split across two GPUs:
bash scripts/vits_minilm/01_train_baselines.sh 0 &    # GPU 0: Linear, MLP, CSA (±STR)
bash scripts/vits_minilm/02_train_ba.sh         1 &   # GPU 1: CLS BA + Token BA (K=128/256/512)

# Once GPU 0's baselines finish:
bash scripts/vits_minilm/03_train_freezealign.sh 0 &  # GPU 0: FreezeAlign (±STR)
```

The scripts just iterate a `(name, config_path)` list and call `PYTHONPATH=. python src/train_alignment.py --config_path <yaml>` for each. Every yaml inherits `configs/default.yaml` and only overrides method-specific knobs.

For a smoke test on CIFAR-10: `PYTHONPATH=. python src/train_alignment.py --config_path configs/dryrun/dryrun_ba.yaml` (≈ 2 minutes).

**Never use** `run_*.py` / `run_*.sh` wrapper scripts — they were deleted; all patches are folded into source (`coco_dataset.py`, `flickr30k_dataset.py`, `csa_trainer.py`).

## What happens on the first run per encoder combo

Layer selection and feature extraction are **automatic**, driven by the config. No manual extraction step. Configs ship without `features.layer_img` / `features.layer_txt` and with `layer_selection.best_only: true` (inherited from default). On the first run the trainer will:

1. Extract all-layer CLS features via `AlignmentTrainer.get_image_features` / `get_text_features` (~10 min for ViT-S, writes to `results/features/`).
2. Run mutual-kNN layer selection over all `(img_layer × txt_layer)` pairs (5,000 random samples, rice-topk ≈ 35) and pick the best.
3. Extract per-layer token features (`pool_img=none`) for the selected layer, with **image-side dedup at extraction** — 414K caption-image rows collapse to 82K unique images, 5× disk and extraction savings. Text keeps all 414K rows.
4. Begin training.

Subsequent runs reuse the same cache and skip steps 1–3. COCO 2014 + ViT-S + MiniLM selects **(layer_img=11, layer_txt=6)** with score 0.2729, rank 1 of 84 pairs — verified empirically on this split. The pair matches the STRUCTURE paper.

## Training defaults (from `configs/default.yaml`)

- Epochs: 1000, early stopping patience 200, seed 42
- Optimizer: AdamW, betas=[0.9, 0.95], weight_decay=1e-4, clip_grad=1.0
- LR: LR finder run per fit, then CosineAnnealingLR with `scheduler_epoch_cycles=50`
- Batch size: 4096 (uniform for CLS and token-level — verified non-OOM in synthetic tests)
- CLIPLoss with temperature 0.05, normalize_latents, warmup_steps 1000
- STRUCTURE regularization (when enabled): `structure_lambda=10.0`, `structure_levels=1`
- `features.image_dedup_extraction: true` (see §feature extraction above)
- `features.img_size: 224` (all experiments; ViT-S → 257 tokens, ViT-L → 257 tokens)

## Alignment methods implemented

| Class | File | Type | Notes |
|---|---|---|---|
| `LinearAlignmentLayer` | `linear.py` | CLS | plain `nn.Linear(input_dim, dim_alignment)` |
| `ResLowRankHead` | `res_low_rank.py` | CLS | STRUCTURE paper's "MLP" — gated residual + low-rank head |
| (CSA, closed-form) | `cca_class.py` + `csa_trainer.py` | CLS | dispatched on `training.cca: true`, not a registered alignment layer |
| `BridgeAnchorAlignmentLayer` | `bridge_anchor.py` | CLS | K learnable anchors, similarity → soft assignment → anchor-weighted sum |
| `BridgeAnchorTokenAlignmentLayer` | `bridge_anchor_token.py` | Token | CAP: anchor × token similarity → softmax over tokens → profile. Supports optional CLS-attention-prior bias (plumbed, extraction deferred). |
| `FreezeAlignAlignmentLayer` | `freeze_align.py` | Token | Maniparambil et al. CVPR 2025. Uses `set_modality('image'\|'text')` routing; image-CLS fallback via `cls_vision_proj`; text-CLS fallback routes through `local_text_proj` as length-1 sequence so `text_proj` always sees `embed_dim` input. |

All layers subclass `BaseAlignmentLayer(input_dim)`, implement `forward(z [, mask])`, and are registered via `@AlignmentFactory.register()`. The trainer instantiates one per modality via the factory and calls `layer.set_modality(...)` if the method defines it.

## Adding a new alignment method

1. Create `src/alignment/<your_layer>.py`.
2. Subclass `BaseAlignmentLayer`, implement `__init__(input_dim, **kwargs)` and `forward(z, mask=None)`.
3. Decorate the class with `@AlignmentFactory.register()`.
4. Add an `import` in `src/alignment/__init__.py`.
5. Write a config under `configs/<method>/<encoder>/` referencing the class by name: `training.alignment_layer_name: "YourLayer"` + `training.alignment_layer_kwargs: {...}`.
6. Add an entry to the appropriate `scripts/<encoder>/0{1,2,3}_*.sh` CONFIGS array.

## Key design decisions / caveats

- **Uniform img_size=224** for all encoders (ViT-S and ViT-L both → 257 tokens at 16×16 patches + CLS). A single token cache serves both CLS and token experiments; CLS runs slice `tokens[:, 0, :]` on the fly.
- **Projector-free BA**: `BridgeAnchorTokenAlignmentLayer` uses `projector_dim: 0` — the anchors themselves are the alignment parameters, no downstream MLP head.
- **`set_modality` pattern** for FreezeAlign: STRUCTURE instantiates one alignment layer per modality with the same class and same kwargs, so image/text components both live on every instance and `set_modality()` routes forward. Other methods auto-detect via mask.
- **Token-level zero-shot eval** (`evaluation.token_level_zero_shot: true`): required for FreezeAlign and Token BA — the CLS-fallback eval path uses the wrong feature distribution and produces random-looking scores if you forget this.
- **CSA sim_dim must be < post-PCA rank**. Our `csa_trainer.py` patches `cca_zoo`'s `MCCA._apply_pca` to use `PCA(n_components=0.999)` (drops zero-variance columns to avoid NaN gradients). On 384-dim ViT-S/MiniLM features this keeps ~364 components; `sim_dim=384` crashes with `scipy.eigh: start=-20, end=363`. Use **`sim_dim=256`** for vits_minilm (verified safe). For 1024-dim ViT-L/RoBERTa, `sim_dim=512` should have plenty of headroom but verify empirically on the first Batch-3 run.
- **FreezeAlign `embed_dim=512`**: the previous `input_dim == embed_dim` constraint on the shared `text_proj` head was relaxed in `src/alignment/freeze_align.py` by routing the CLS fallback through `local_text_proj` as a length-1 sequence. All 4 FA configs now set `embed_dim: 512`.
- **STRUCTURE caches all features to disk first**, then trains alignment only — no encoder forward during training. Each epoch is very fast; 1000 epochs is the default and early stopping usually kicks in well before that.

## Known issues / workarounds (now in source, no wrappers needed)

- `CocoCaptionDataset.load_image` / `Flickr30kDataset.load_image`: returns `PIL.Image` (was `torch.Tensor`) so the timm transform pipeline's `ToTensor()` runs on the right input type. See `src/core/src/datasets/downstream_tasks/{coco,flickr30k}_dataset.py`.
- `cca_zoo.linear._mcca.MCCA._apply_pca`: monkey-patched at `csa_trainer.py` import time to use `PCA(n_components=0.999)` — drops zero-variance columns in the whitening step.
- Wrapper scripts (`run_with_totensor_fix.py`, `run_csa_fix.py`, `run_dryrun.sh`) are **deleted**. Don't recreate them.
- **Dedup-at-extraction gate**: auto-disabled when `training.drop_duplicates=false`, `n_dup_samples != 1`, or `n_random_subsample_train` is set (dryrun paths). Doesn't currently distinguish eval-loader vs train-loader — don't call `get_image_features(pool=none)` on the COCO val loader without reading the fit() assumption about shared row counts first.

## Evaluation

Every run evaluates:
- **Zero-shot classification**: CIFAR10, CIFAR100, STL10, MNIST (default set in most configs; default.yaml has a much larger 22-dataset list we can enable when cache is warm)
- **Retrieval**: COCO val (I2T/T2I R@1/5/10, MAP@k), Flickr30k if in the config

Token-level methods set `evaluation.token_level_zero_shot: true` to use the CAP path for the template embedding (otherwise eval scores look random). Retrieval always uses the trained alignment layer's output.

For post-hoc re-evaluation of a trained checkpoint on a different dataset list, use `rerun_eval.py`:
```bash
PYTHONPATH=. python rerun_eval.py \
    --config_path configs/<same config used for training>.yaml \
    --ckpt results/alignment-.../checkpoint-epoch<N>.pth \
    --label my_rerun \
    --zs dtd,flowers,gtsrb --rt flickr30
```

## Data

- **COCO 2014** at `data/COCO/` — real 2014 split (82,783 train / 40,504 val), real 2014 JSON annotations. Downloaded fresh on Apr 14 after discovering the old `data/COCO/` was 2017-masquerading-as-2014 via symlinks. Old setup archived in `archive/2026-04-14_coco2017_vits/`.
- **Flickr30k** at `data/flickr30k/` — cleaned single-directory layout after the two-level symlink chain was flattened Apr 14. Karpathy split files (`train.txt`/`val.txt`/`test.txt`) + pipe-delimited `results.csv` + `images/` (29,783 / 1K / 1K).
- **ImageNet**: only `val` (~6 GB) + `LOC_synset_mapping.txt` are needed for zero-shot eval. `train` (~130 GB) is **not required** — `src/dataset_preparation/data_utils.py::get_datasets("imagenet", ...)` handles its absence gracefully: if `data/imagenet/train/` is missing or empty, the train `ImageFolder` is set to `None` and only the val split is returned. On Server B, provision val only.
- **Auto-downloaded zero-shot datasets** (via torchvision): DTD, Flowers102, GTSRB, Country211, STL10, CIFAR10/100, MNIST, SUN397, Pets, Cars, Aircraft, Birdsnap, Food101, Caltech101, EuroSAT, Resisc45, UCF101, Kinetics700, PCam, HatefulMemes, SST, KITTI, FER2013, Clevr — not all tested; the 4-dataset subset in most configs (`cifar10/stl10/cifar100/mnist`) is the fast eval loop.

## Archive

- `archive/2026-04-14_coco2017_vits/` — all Batch-1 outputs (COCO 2017 feature caches, wandb, logs, checkpoints) plus the old run scripts (`run_overnight_gpu{0,1}.sh`, `run_rerun_eval_gpu1.sh`). 26 GB total. Not comparable to Batch 2 (different data split) — keep for reference only.
- `archive/orig_structure_build/` — obsolete STRUCTURE build system from the initial commit: `Dockerfile`, `.dockerignore`, `Makefile`, `common.mk`, `requirements.txt`. Superseded by `docker/Dockerfile.structure` + `docker/requirements.structure.txt`. 8 KB.
- `configs/_reference_structure/` — original STRUCTURE configs from the initial commit (ablations/clip/csa/data/losses_lin/losses_mlp/metrics + original default.yaml). Frozen, never run.

## Development rules

- **Always update `PROJECT_LOG.md`** after every non-trivial task (new date-stamped section for separate work days, append for same-day continuations).
- **Test with a dryrun config before kicking off the full experiment queue**: `configs/dryrun/dryrun_*.yaml` run in ~2 minutes on CIFAR-10.
- **Use wandb** for metrics — runs are logged to project `representation-alignment` (or `representation-alignment-CSA` for CSA runs).
- **Never create** wrapper scripts or monkey-patches in a new file. If you need to patch library behavior, do it at the source-file import where the call happens (pattern: `csa_trainer.py` patching `cca_zoo` at import).
- **Never commit** to `archive/*` — it's gitignored. Copy files into it for safekeeping.
- When adding a feature, also update the relevant docs (this file, `IMPLEMENTATION.md`, `PROJECT_LOG.md`, and if it affects configs also `configs/README.md`). Don't let the docs rot.
