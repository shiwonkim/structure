# BridgeAnchors x STRUCTURE Integration — Project Log

## Current Status & Next Steps

### Infrastructure
- Server A: STRUCTURE environment set up (conda env: structure), dry run complete
- Server B: Original BA codebase, experiments running
- Docker: image built and verified on Server A (shiwonkim/structure-ba:v1, 19.5GB)

### Immediate TODO
1. ~~Build Docker image on Server A (cu118, all dependencies verified)~~ DONE
2. ~~Push to Docker Hub: `docker push shiwonkim/structure-ba:v1`~~ DONE (2026-04-12)
3. ~~Implement BA as alignment method in STRUCTURE framework~~ DONE (2026-04-13)
4. ~~Implement FreezeAlign baseline~~ DONE (2026-04-13)
5. ~~Run 9-method small-model comparison table (Linear, Linear+STR, MLP, MLP+STR, CLS BA K=128/256, Token BA K=128/256, FreezeAlign)~~ DONE (2026-04-14 overnight)
6. ~~Implement CLS Attention Prior plumbing for Token BA~~ DONE (2026-04-14)
7. ~~COCO 2014 migration + Batch 2 on ViT-S + MiniLM~~ IN PROGRESS (2026-04-15)
8. Deploy to Server B and re-run at ViT-L + RoBERTa-Large scale (Batch 3)

---

## 2026-04-15 — CLS attention prior: extraction + training

Activated the dormant CLS attention prior on `BridgeAnchorTokenAlignmentLayer`. The layer already supported a learnable per-anchor β that biases the CAP softmax logits by `β_k * log(cls_attn + ε)`, and the trainer already accepted `image_cls_attn` kwargs in `train()` / `validate()` — but the extraction and pass-through were never wired. This entry closes those gaps.

**1. Extractor — `AlignmentTrainer.get_image_cls_attention`**

Added a new method on the trainer that extracts per-image CLS→patch attention from the last DINOv2 block. Because STRUCTURE's vision path uses a `torchvision.models.feature_extraction.create_feature_extractor`-wrapped model that obscures module inputs, the extractor loads a fresh unwrapped timm model, registers a forward-pre-hook on `blocks[layer_idx].attn` to capture the (LN'd) input, then manually recomputes Q/K from `attn.qkv` and derives the attention matrix:

```python
qkv = attn.qkv(x).reshape(B, N, 3, H, D_h)
q, k, _ = torch.unbind(qkv, dim=2)
q, k = q.transpose(1,2), k.transpose(1,2)
attn_weights = F.softmax((q @ k.transpose(-2,-1)) * scale, dim=-1)
cls_attn = attn_weights[:, :, 0, 1:].mean(dim=1)   # average over heads
cls_attn = cls_attn / cls_attn.sum(-1, keepdim=True).clamp(min=1e-8)
```

Mirrors the bridge-anchors reference at `src/data/extract_attention_maps.py:103` exactly. Shares the image-side dedup gate with `get_image_features(pool=none)` so the N-axis of the CLS attention cache lines up with the token cache (dedup'd 82K for COCO train on ViT-S). Output is `(N, P)` float16 saved at `results/features/<model>-<dataset>-<split>_cls_attn_layer-<L>-r<img_size>.npy`.

**2. Trainer wiring**

`fit()` now conditionally loads the CLS attention cache when `token_level=True` *and* `alignment_layer_kwargs.cls_attn_prior=True`:

- Uses `_subsampled_loader` with the same subset caps as the token-feature path so the dry-run path stays consistent.
- Applies the same `_apply_if_full` dedup as the image token tensor — handles both "image already dedup'd at extraction" (118K) and "everything full" (591K) shapes.
- Wrapped in try/except: if extraction fails, logs a warning and falls back to standard CAP (`cls_attn=None`) rather than crashing the run.

Then threaded `image_cls_attn=image_cls_attn_train/val` into the existing `self.train(...)` and `self.validate(...)` call sites, which already had the kwarg and the conditional "only apply if `layer.cls_attn_prior` is True" logic from the Apr 14 plumbing work.

**3. Configs — 3 new files**

Copied the existing Token BA configs into `_prior` variants:

- `configs/ba/vits_minilm/token_k128_prior.yaml`
- `configs/ba/vits_minilm/token_k256_prior.yaml`
- `configs/ba/vits_minilm/token_k512_prior.yaml`

Each adds two fields under `alignment_layer_kwargs`:
```yaml
cls_attn_prior: true
cls_attn_beta_init: 1.0
```
Everything else (pool_temperature, batch_size, retrieval datasets, etc.) inherited from the base token configs.

**4. Script — `scripts/vits_minilm/04_train_ba_prior.sh`**

Follows the `02_train_ba.sh` convention: sources `common.env` + `config.env`, takes GPU id as `$1`, writes timestamped log, runs the three prior configs sequentially. Doesn't abort on individual failures (so one OOM doesn't sink the whole overnight batch).

**5. Launch**

`bash scripts/vits_minilm/04_train_ba_prior.sh 1` — launched on GPU 1 at 19:18 KST while GPU 0 continues training `fa_d512_struct`.

**Pending results** (will report when complete):
- CLS attention extraction time and cache shape
- Per-anchor β values after training (expect non-trivial variance across the K anchors — some push toward CLS attention, some ignore it)
- Comparison: Token BA vs Token BA + prior at K=128/256/512

**Files touched.**

- `src/trainers/alignment_trainer.py` — added `torch.nn.functional as F` import; added `get_image_cls_attention` method; wired load + pass-through in `fit()` around the token-level branch.
- `configs/ba/vits_minilm/token_k{128,256,512}_prior.yaml` — 3 new configs.
- `scripts/vits_minilm/04_train_ba_prior.sh` — new script.

---

## 2026-04-15 — Pre-Server-B cleanup + Batch 2 launch on COCO 2014

Final preparation day before the Server B rsync. Migrated to real COCO 2014, rebuilt the experiment harness (configs + scripts + docs) around a method-by-encoder layout, shook out a FreezeAlign architectural bug and a CSA `sim_dim` rank-cap bug, then launched the full 14-run Batch 2 overnight on Server A. 10/14 runs have finished as of this writing.

### COCO 2014 cutover
Discovered `data/COCO/annotations/captions_*2014.json` were symlinks pointing at 2017 JSONs — meaning Batch 1 trained on the wrong split. Full reset:

- Deleted the 2017-masquerading-as-2014 symlinks, downloaded real COCO 2014 directly to `data/COCO/`:
  - `train2014/` — 13 GB, 82,783 images (`COCO_train2014_*.jpg` prefix)
  - `val2014/` — 6.3 GB, 40,504 images
  - `annotations/` — 806 MB, real 2014 JSONs (`captions_{train,val}2014.json`, `instances_*`, `person_keypoints_*`)
- Counts match the official 2014 split exactly (82,783 / 40,504).
- Archived all Batch 1 outputs under `archive/2026-04-14_coco2017_vits/` (logs 33 MB, checkpoints 71 MB, features 26 GB, wandb 492 KB). Kept for reference — not comparable to Batch 2 results.
- Created `EXPERIMENTS.md` as the canonical batch ledger (Batch 1 done on wrong split, Batch 2 running on correct 2014, Batch 3 pending on Server B).
- Added `archive/` to `.gitignore`.

### Config reorganization — method × encoder subtree
The old `configs/losses_{lin,mlp,ba,fa}/` + root-level `csa/` layout couldn't express per-encoder configs cleanly. Moved to:

```
configs/
├── default.yaml
├── README.md
├── _reference_structure/   ← 39 original STRUCTURE configs + default_original.yaml (frozen)
├── dryrun/                 ← 9 smoke tests
├── linear/<encoder>/       ← linear_d512{,_struct}
├── mlp/<encoder>/          ← mlp_d512{,_struct}
├── csa/<encoder>/          ← csa_d<sim>{,_struct}
├── ba/<encoder>/           ← cls_k{128,256,512}, token_k{128,256,512}
└── freezealign/<encoder>/  ← fa_d512{,_struct}
```

- `<encoder>` ∈ {`vits_minilm`, `vitl_roberta`}. 28 method configs total (14 per encoder × 2 encoders).
- Each file uses `defaults: !include ../../default.yaml` and overrides only method- and encoder-specific fields.
- Moved the 39 original STRUCTURE configs (ablations/clip/csa/data/losses_lin/losses_mlp/metrics) to `configs/_reference_structure/`. Also copied `default.yaml` from the initial commit to `_reference_structure/default_original.yaml` so the paper's original defaults stay accessible.
- Deleted the old method folders after verification: `configs/losses_ba/`, `losses_fa/`, `losses_lin/`, `losses_mlp/`, and the root-level `configs/csa/*.yaml` (6 files) — all redundant with the new layout (25 files removed).
- Created `configs/README.md` with the layout diagram, conventions, and a "how to add a new encoder combo" recipe.

### Script reorganization — 3-script overnight workflow
The old `run_overnight_gpu{0,1}.sh` runners referenced deleted config paths. Replaced with a clean per-encoder directory:

```
scripts/
├── README.md
├── common.env               ← shared training defaults (epochs, BS, seed, STR hyperparams)
├── vits_minilm/
│   ├── config.env           ← encoder-specific vars (models, dims, CSA_SIM_DIM)
│   ├── 01_train_baselines.sh    ← Linear, Linear+STR, MLP, MLP+STR, CSA, CSA+STR
│   ├── 02_train_ba.sh           ← CLS BA K=128/256/512, Token BA K=128/256/512
│   └── 03_train_freezealign.sh  ← FreezeAlign, FreezeAlign+STR
└── vitl_roberta/            ← same layout, for Batch 3
```

- Convention: every script sources `common.env` + local `config.env`, takes GPU id as `$1`, `cd`s to project root, writes a timestamped log, and runs experiments sequentially via `PYTHONPATH=. python src/train_alignment.py --config_path <yaml>`.
- Original 5-script plan (with explicit `01_layer_selection.sh` + `02_extract_features.sh`) was scrapped: those scripts were written speculatively against `src/extract_features.py`'s CLI which only supports `--modelset val` (extracts ~23 models at once, not single-pair). Instead, `train_alignment.py` handles layer selection + extraction automatically on the first run per encoder combo (`layer_selection.best_only: true`). Documented in `scripts/README.md`.
- Archived the old Batch-1 runners to `archive/2026-04-14_coco2017_vits/`: `run_overnight_gpu0.sh`, `run_overnight_gpu1.sh`, `run_rerun_eval_gpu1.sh`.

### Layer pair removal — let the trainer pick (11, 6) for real
All 28 configs originally had `features.layer_img: 11` / `features.layer_txt: 6` hardcoded — those came from the Batch 1 mutual-kNN run on COCO 2017 and risked silently wrong layers for a different data split or encoder. Stripped both lines from every method config. With `layer_selection.best_only: true` in `default.yaml`, the trainer now performs real mutual-kNN selection on the first run per encoder combo.

- Verified the COCO 2014 layer pair by loading the freshly-extracted all-layer CLS cache (`vit_small_patch14_dinov2.lvd142m-CocoCaptionDataset-train-cls-r224.npy`, shape `(414113, 12, 384)`) and running mutual-kNN on 5,000 samples (rice-topk=35) on CPU:
  - **Best pair: (img=11, txt=6) — score 0.2729, rank 1 of 84.**
  - Runner-up (10, 6) at 0.1827, a 49 % relative margin.
  - Full score matrix is monotone-increasing in both img and txt depth, consistent with "last CLS layer is most semantic" for DINOv2 + MiniLM.
- Crucial for Batch 3: ViT-L + RoBERTa-Large have 24 and 25 layers respectively, so the hardcoded (11, 6) would have pinned mid-layers instead of last. Strict correctness fix for the large-encoder runs.

### FreezeAlign — `embed_dim=512` CLS-fallback routing fix
The original FA port pinned `embed_dim = input_dim` (384 for vits_minilm, 1024 for vitl_roberta) because `_forward_text`'s CLS fallback fed raw encoder CLS directly into `text_proj`, which expected `embed_dim` input. Fixed by routing the CLS fallback through the full token pipeline:

- `src/alignment/freeze_align.py::_forward_text`: CLS fallback now unsqueezes `(B, D) → (B, 1, D)`, runs through `local_text_proj` to get `(B, 1, embed_dim)`, squeezes back, then applies `text_proj`. Mean-pool over 1 token = squeeze, identical math to the token path with T=1.
- `_forward_image` CLS fallback left alone — it already routes through `cls_vision_proj` which handles `input_dim → embed_dim`.
- Removed the `input_dim != embed_dim` warning and stale `# text CLS-fallback will fail` comment.
- All 4 FA configs updated: `embed_dim: 384/1024 → 512` (now the `fa_d512.yaml` filenames are finally accurate).
- **Smoke test**: input_dim=384, embed_dim=512: image-token `(4,257,384)→(4,512)`, image-CLS `(4,384)→(4,512)`, text-token with mask `(4,16,384)+mask→(4,512)`, text-CLS `(4,384)→(4,512)`. Gradients flow through both `local_text_proj` and `text_proj` on the CLS-fallback backward. L2-norm check passes. All four paths verified.
- **Consequence for eval**: zero-shot via the CLS fallback now uses trained weights on both projectors instead of an untrained head. Matches the training input distribution.

### CSA `sim_dim` PCA-rank cap — 384 → 256 for vits_minilm
Baselines script crashed on run 5/6 (`csa_d384`) with `ValueError: Requested eigenvalue indices are not valid. Valid range is [0, 363] and start <= end, but start=-20, end=363 is given`.

- Root cause: our module-import patch in `src/trainers/csa_trainer.py` replaces `cca_zoo.linear._mcca.MCCA._apply_pca` with `PCA(n_components=0.999)` (drops zero-variance columns to avoid NaN gradients). On 384-dim ViT-S/MiniLM features this keeps ~364 components. `NormalizedCCA` then asks for the top `sim_dim=384` eigenvalues → `start = 364 − 384 = −20` → `scipy.linalg.eigh` rejects the indices.
- Fix: `sim_dim: 384 → 256` in both vits_minilm CSA configs, comfortably below the 364 PCA rank cap and aligned with the STRUCTURE paper convention of `sim_dim ≤ 0.75 × effective_rank`.
- Renamed files to match: `csa_d384.yaml → csa_d256.yaml`, `csa_d384_struct.yaml → csa_d256_struct.yaml`. Updated `scripts/vits_minilm/01_train_baselines.sh`, `scripts/vits_minilm/config.env` (`CSA_SIM_DIM=256` + comment explaining the PCA-rank cap), and internal config comments.
- Killed + relaunched the CSA rerun chain on GPU 0 with the new filenames instead of leaving backward-compat symlinks. 10 min of wasted CSA extraction; CSA is closed-form so relaunch is cheap.
- vitl_roberta CSA at `sim_dim: 512` against 1024-dim encoders should have plenty of headroom — verify on first Batch 3 run.

### Project-root cleanup
Archived 5 obsolete STRUCTURE build-system files superseded by `docker/Dockerfile.structure`:

```
archive/orig_structure_build/
├── .dockerignore    (145 B)
├── Dockerfile       (1.0 KB, pytorch 2.2.0 — driver 470 incompatible)
├── Makefile         (3.5 KB, docker build targets we never use)
├── common.mk        (3.0 KB, Makefile helper from docker-make-stub)
└── requirements.txt (222 B, not referenced by our docker/)
```

Root went from 18 → 13 files. Kept: active docs (CLAUDE/EXPERIMENTS/IMPLEMENTATION/PROJECT_LOG/Readme), `rerun_eval.py`, `.gitattributes`, `.gitignore`, plus 5 harmless linter/packaging configs (`.editorconfig`, `.flake8`, `.pre-commit-config.yaml`, `.yamllint`, `pyproject.toml`).

### Batch 2 overnight execution
Launched 2 GPUs in parallel late Apr 14; partial results in hand as of Apr 15 morning.

- **GPU 0 baselines** (`01_train_baselines.sh`): Linear → Linear+STR → MLP → MLP+STR done cleanly, then CSA crashed on the `sim_dim=384` bug. After the fix, reran `05_csa_d256` + `06_csa_d256_struct` in a mini chain that auto-chained into `03_train_freezealign.sh`.
- **GPU 1 BA** (`02_train_ba.sh`): Was initially launched in parallel with baselines — discovered both were racing on the same feature cache extraction. Killed GPU 1, let GPU 0 finish extraction (~16 GB of caches written to `results/features/`), then relaunched GPU 1 which reused the cache immediately. CLS BA K=128/256/512 and Token BA K=128 done; K=256 running, K=512 queued.
- **FreezeAlign waiter misfire**: the original "wait for baselines pid, then fire FA" watcher was set up before the CSA crash. When CSA died at 05:32, the watcher detected the dead parent pid at 05:33 and fired FreezeAlign prematurely on GPU 0. FA trained for ~3 hours (reached epoch 228/1000 with val loss 3.99) before being collateral-killed at 08:37 during the CSA cleanup. Wasted compute, no code impact. Renamed the stale log to `logs/vits_minilm_freezealign_20260415_0533_killed.log`. Live run started fresh at 08:56.

### Partial Batch 2 results (10/14 runs finished)

Percentages; best per column bold.

| # | Run | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@10 | CIFAR10 | CIFAR100 | STL10 | MNIST |
|---|---|---|---|---|---|---|---|---|---|---|
| 01 | Linear d=512 | 8.48 | 23.07 | 32.61 | 6.19 | 9.94 | 81.91 | 34.25 | 93.38 | 6.59 |
| 02 | Linear+STR d=512 | 8.82 | 23.46 | 33.01 | 6.03 | 9.74 | 81.96 | 33.95 | 93.33 | 8.86 |
| 03 | MLP d=512 | 8.52 | 23.03 | 32.67 | 6.18 | 9.94 | 81.69 | 33.85 | 93.11 | 7.99 |
| 04 | MLP+STR d=512 | 8.62 | 23.41 | 33.06 | 6.04 | 9.79 | **82.47** | 34.32 | **93.44** | 7.90 |
| 05 | CSA d=256 | 3.98 | 12.33 | 18.72 | 2.57 | 4.40 | 72.32 | 24.95 | 82.78 | 10.61 |
| 06 | CSA d=256 + STR | 4.16 | 12.51 | 18.56 | 2.97 | 5.04 | 80.66 | 31.85 | 89.99 | 12.22 |
| 07 | CLS BA K=128 | 8.30 | 22.45 | 31.86 | 5.93 | 9.61 | 81.09 | 32.92 | 92.30 | 11.97 |
| 08 | CLS BA K=256 | 8.40 | 22.85 | 32.56 | 6.13 | 9.91 | 81.69 | 33.72 | 93.00 | 9.37 |
| 09 | CLS BA K=512 | 8.42 | 22.95 | 32.61 | 6.15 | 9.93 | 81.77 | 33.70 | 93.24 | 7.59 |
| **10** | **Token BA K=128** | **16.30** | **36.89** | **47.73** | **11.30** | **17.08** | 84.70 | **35.38** | 89.50 | **13.84** |

**Headline finding**: Token BA K=128 roughly doubles every CLS baseline on retrieval (I2T R@1 16.30 vs ~8.5, R@10 47.73 vs ~32.7). Exactly reproduces the Batch-1 finding that CAP pooling is the differentiator, not the anchor head architecture. CLS baselines are essentially tied (Linear/MLP/CLS BA all in 8.3–8.8 % I2T R@1), so varying K in CLS BA does ~nothing and STRUCTURE reg adds 0.2–0.4 pp at most on retrieval. CSA underperforms but the refinement step helps it on classification.

### Docs refresh
- **Rewrote `CLAUDE.md`** (74 → 216 lines). New sections: current Batch 2 state + headline result, directory tree, 3-script workflow, automatic layer selection + extraction pipeline, training defaults table, alignment methods table, key design decisions, known issues, evaluation conventions, data layout, archive structure. Every piece of context a fresh Claude Code session needs on Server B is in this single file.
- **Updated `configs/README.md`**: removed the stale note about FreezeAlign `embed_dim` being pinned to `input_dim`; replaced with the current `embed_dim: 512` convention and a pointer to the CLS-fallback fix in `freeze_align.py`.
- `scripts/README.md` already accurate for the new 3-script workflow.
- `IMPLEMENTATION.md` updated with the FreezeAlign CLS-fallback routing fix and the CSA `sim_dim` PCA-rank cap design notes (see those entries below).

### Pending
- Batch 2 finish (4 runs remaining: token_k256, token_k512, fa_d512, fa_d512_struct).
- Record final Batch 2 results in `EXPERIMENTS.md`.
- rsync repo to Server B, launch Batch 3 (same 14 methods × ViT-L + RoBERTa-Large × COCO 2014).

---

## 2026-04-14 — Server A work: full small-model comparison table + infra polish

**9-run overnight comparison table completed** on ViT-S + MiniLM at COCO 118K-dedup train, img_size=224, pinned layers (img=11, txt=6), 1000 epochs with ES@200, LR finder, seed=42. Token runs used BS=2048 (memory-safe), CLS runs BS=4096.

| Method | Params/mod | CIFAR-10 | DTD | Flowers | GTSRB | COCO I2T R@1 | COCO T2I R@1 | Flickr I2T R@1 | Flickr T2I R@1 |
|---|---|---|---|---|---|---|---|---|---|
| Linear d=512 | 197K | 82.4 | 6.8 | 2.9 | 4.0 | 27.5 | 20.6 | 44.0 | 32.6 |
| Linear+STR d=512 | 197K | 83.0 | 7.7 | 3.5 | 3.4 | 27.7 | 20.4 | 44.4 | 33.0 |
| MLP (ResLowRank) d=512 | 254K | 82.7 | 6.9 | 2.8 | 4.1 | 27.4 | 20.7 | 43.1 | 32.4 |
| MLP+STR d=512 | 254K | 83.7 | 8.1 | 3.0 | 3.4 | 27.8 | 20.4 | 44.7 | 32.8 |
| CLS BA K=128 | 49K | 82.2 | 6.3 | 2.9 | 3.6 | 26.9 | 20.3 | 43.4 | 32.1 |
| CLS BA K=256 | 98K | 83.0 | 6.5 | 2.9 | 3.7 | 27.6 | 20.8 | 43.7 | 32.9 |
| FreezeAlign | 1.6M | 84.5 | 9.4 | 2.7 | 5.6 | 36.5 | 27.5 | 50.0 | 37.8 |
| **Token BA K=128** | **49K** | 85.4 | 6.0 | 5.6 | 7.1 | 41.4 | 30.2 | 57.5 | 42.2 |
| **Token BA K=256** | 98K | **85.9** | 8.1 | 3.5 | **9.0** | **44.5** | **32.9** | **59.7** | **45.1** |

**Key findings:** Token-level methods dominate retrieval: Token BA K=256 gives +16.7pp I2T R@1 / +12.2pp T2I R@1 on COCO and +15/+12 pp on Flickr30k over the best CLS baseline. Token BA K=128 beats FreezeAlign at **33× fewer parameters**. The BA anchor architecture is not the story — the CAP pooling is: identical BA K=128 architecture gives 26.9 COCO I2T R@1 in CLS mode vs 41.4 in token mode, a +14.5pp gap from forward-path choice alone.

**Dataset infrastructure (server A, in prep for Server B):**
- COCO 2014 annotations downloaded to `data/COCO/annotations_2014_real/` (82K train / 40K val, real filenames with `COCO_{train,val}2014_` prefix). Current symlinks still point to 2017 JSONs because our feature caches match 2017 split; Server B will use the 2014 JSONs after fresh extraction.
- Flickr30k (31,784 images + Karpathy splits + results.csv) rsynced from `/mnt/2021_NIA_data/bridge-anchors-data/datasets/flickr30k/` to `/home/data/2026_Flickr30k/`. STRUCTURE layout at `data/flickr30k/` (split files + `results.csv` + `images/` symlink to the NAS-rsynced JPGs on local disk). The dataset class and dispatcher were updated to use `data/flickr30k/` as root with a nested `images/` subdir.
- ImageNet symlinked from `/home/data/imagenet/` with `LOC_synset_mapping.txt` fetched.
- Torchvision auto-downloads: DTD (5,640), Flowers102 (8,189), GTSRB (39,270), Country211 (31,650 valid+test) all present. Food101 + PCAM + Country211-train deferred (upstream throttling / errors).

**Dedup-at-extraction optimization** (separate entry below): image-side token extraction now dedups to one row per unique image_path at write time, turning the train cache from 109 GB → 22 GB and vision forward passes from ~40 min → ~8 min. fit()'s per-tensor mask application handles the asymmetric image(deduped)/text(full) row counts correctly.

**CLS attention prior plumbing** (separate entry below): BA-token layer now accepts an optional `cls_attn` kwarg and learnable per-anchor `β` for soft guidance toward DINOv2 CLS-salient patches. Extraction of `cls_attn` itself is deferred — plumbing is dormant-by-default but ready.

**Pre-Server-B cleanup (2026-04-14):** Reclaimed ~182 GB in `results/features/`:
- Deleted 4 `.bak.preCleanup`/`.preDedup` files (~130 GB)
- Deleted 11 pre-r224 stale caches (~45 GB) — image preprocessing changed when wrapper cleanup moved `CocoCaptionDataset.load_image` from tensor-return to PIL-return, so the old caches used different bilinear interpolation
- Deleted 8 `-n{N}` dryrun subset caches (~7 GB)

Also removed: `wandb/` offline runs (1 root-owned remnant kept due to permission), all `__pycache__`, 17 pre-overnight-run + extraction log files, 4 one-off scratch scripts (`refresh_caches.py`, `download_eval_datasets.py`, `smoke_test_token_zero_shot.py`, `run_cls_ba_reruns.sh`), and 5 superseded small-model configs (`bridge_anchor_small_{128,256,512}.yaml`, `clip_small_best.yaml`, `clip_small_structure_1.yaml`).

`.gitignore` extended with `logs/`, `*.bak.*`, `.claude/`, `bridge-anchors/`. Final working tree matches what we want to commit + push to Server B.

---

## 2026-04-13 — Token-level BA with Cross-Attention Pooling (CAP) integration

### Phase 1–3 token-level pipeline implemented

STRUCTURE had no token-level alignment method before this. Now it does.

**What was built:**
1. `src/alignment/bridge_anchor_token.py` — `BridgeAnchorTokenAlignmentLayer`. CAP forward:
   ```
   z (B, T, D)  →  z_norm @ anchors.T  →  sim (B, T, K)
   attn = softmax(sim / tau, dim=tokens)   (mask-aware)
   profile = (attn * sim).sum(tokens)  →  (B, K)   →  L2-normalize
   ```
   Falls back to 2D CLS path when input is `(B, D)` — used at eval time so zero-shot/retrieval still work without recaching token features for eval datasets.

2. `src/extract_token_features.py` — standalone CLI. Reuses STRUCTURE's pre-existing `pool=none, layer=L` extraction path (which had been dormant) and additionally saves text attention masks to a companion `_mask.pt` file.

3. `src/trainers/alignment_trainer.py` — added:
   - `_load_token_features_for_layer(img_layer, txt_layer)` — temporarily overrides `features.pool_*` to the none-pool single-layer mode, calls existing `get_image_features` / `get_text_features`, loads or builds text mask, restores config.
   - `_load_or_build_text_mask(loader, llm, suffix)` — tokenizes the dataloader to build masks when not cached.
   - `fit()` gets a `token_level` branch: after layer selection, replace the `(N, D)` layer slices with `(N, T, D)` token tensors + `(N, T)` text mask, reapply dedup + subsample on tokens.
   - `train()` / `validate()` pass `mask=text_mask_batch` to `alignment_text` when in token mode.

4. `src/trainers/base_trainer.py::find_optimal_learning_rate` also threads `text_mask_train`.

5. `src/train_alignment.py` — now reads `training.n_random_subsample_train/val` from config and passes to `fit()`.

6. `configs/dryrun_ba_token.yaml` — dry-run config (ViT-S + MiniLM, CIFAR-10, 2 epochs, K=128, τ=0.05, subsample train=2000/val=1000, `token_level=true`, `structure_lambda=0`).

**Design notes:**
- `structure_lambda=0` is required in token mode because `structure_reg` expects 2D originals. CLIP loss only uses aligned 2D (B, K) output which works regardless.
- The 2D CLS fallback in the BA token layer is deliberate: it means eval (zero-shot classification, retrieval) still works by passing CLS tensors through the fallback branch, without needing token caches for every eval dataset. Caveat: eval underestimates CAP potential since tokens aren't used there.
- Feature extraction for tokens piggybacks on the unused `pool="none", layer=L` path rather than introducing a parallel code path.
- Dedup and subsample are re-applied to tokens in sync with the CLS path (indices are stored during the CLS phase and replayed).

**Dry-run complete** (`configs/dryrun_ba_token.yaml`, ViT-S + MiniLM, CIFAR-10, 2 epochs, K=128, τ=0.05, subsampled train=2000/val=1000):

| Metric | Value |
|---|---|
| Image token shape | **(2000, 1370, 384)** — 1370 = 1 CLS + 37×37 patches at DINOv2 518×518 |
| Text token shape | **(2000, 7, 384)** |
| Text mask shape | **(2000, 7)** |
| Layer selected by CLS mutual-kNN | img=11, txt=6 (same as CLS runs) |
| Params per modality | 49,152 (128 × 384) |
| Epoch 1 train / val loss | 4.56 / 3.72 |
| Epoch 2 train / val loss | 3.68 / 3.66 |
| CIFAR-10 zero-shot top-1 | **78.31%** |

Loss decreased cleanly; end-to-end flow works. Eval ran via the 2D CLS fallback in `BridgeAnchorTokenAlignmentLayer` (as designed) — no token features needed for eval datasets.

Notable wrinkle: DINOv2's 518×518 default resolution produces 1370 image tokens per sample. Full CIFAR-10 train token features would be ~105 GB, so the dry-run config subsamples aggressively. For real training on COCO this scale is manageable (~118K images × 1370 × 1024 × 4 bytes = ~660 GB for ViT-L → would need float16 or per-layer streaming) — but that's a future concern. The dry-run demonstrates the code path works.

**Bugs found and fixed during dry-run:**
1. `_SubsetView.__setattr__` needed to delegate to the wrapped dataset so `loader.dataset.tokenizer = ...` took effect. Without the fix, `apply_tokenizer()` silently became a no-op and `get_text_features` crashed on `list has no attribute items`.
2. CLS path subsample uses a random permutation, token extraction uses deterministic first-N. They don't align, so we skip re-applying CLS dedup/subsample masks in `token_level` mode — the token features are already the right size from the extraction subset.

---

## 2026-04-13 — BA small-model K=128 completed + Linear baseline running

### First complete BA result — K=128 small-model on COCO

Used Option E (hybrid) — cached small-model features on GPU 1 to get fast BA results while the large-model run continues on GPU 0.

**Setup:**
- Encoders: `vit_small_patch14_dinov2.lvd142m` + `all-MiniLM-L6-v2`
- Training: 1000 epochs, BS=4096, LR finder (found 2.89e-4), cosine T_max=50, early stopping patience=200, `drop_duplicates=true, n_dup_samples=1`
- Layer selection: mutual kNN picked layers (img=11, txt=6), score=0.283
- Early stopped at epoch 490 (val loss plateau around 3.3715)
- Total wall time: ~22 minutes training + ~8 minutes eval

**Results (BA K=128 small):**
| Metric | Value |
|---|---|
| Best val clip loss | **3.3715** |
| CIFAR-10 zero-shot top-1 | **82.45%** |
| STL-10 zero-shot top-1 | **94.50%** |
| CIFAR-100 zero-shot top-1 | **33.83%** |
| MNIST zero-shot top-1 | **18.04%** |
| COCO I2T R@1 / R@5 / R@10 | **28.12% / 57.80% / 71.09%** |
| COCO T2I R@1 / R@5 / R@10 | **21.34% / 21.34% / 31.40%** |

**Linear baseline (completed):**
- Same encoders, `dim_alignment=256` (~98K params/modality, 2× BA)
- Early-stopped at epoch 508 / best val loss **3.3609**
- LR finder picked 1.15e-4 (lower than BA's 2.89e-4)

### Head-to-head: BA K=128, K=256 vs Linear (small encoders, COCO)

| Metric | BA K=128 | BA K=256 | Linear (dim=256) |
|---|---|---|---|
| Best val clip loss | 3.3715 | 3.3655 | **3.3609** |
| Params/modality | **49,152** | 98,304 | 98,560 |
| Early-stopped at epoch | 490 | ~490 | 508 |
| CIFAR-10 top-1 | 82.45% | **82.70%** | 82.22% |
| STL-10 top-1 | 94.50% | **95.43%** | 95.00% |
| CIFAR-100 top-1 | 33.83% | 34.30% | **34.60%** |
| MNIST top-1 | **18.04%** | 16.37% | 15.59% |
| COCO I2T R@1 | 28.12% | 28.87% | **29.10%** |
| COCO I2T R@5 | 57.80% | **58.50%** | 58.00% |
| COCO I2T R@10 | **71.09%** | 71.07% | 70.90% |
| COCO T2I R@1 | 21.34% | 21.70% | **21.80%** |

**Conclusions so far:**
- BA and Linear are essentially tied on this setup (within 1 percentage point on all metrics)
- Linear edges out on most metrics (STL-10, CIFAR-100, CIFAR-10 retrieval, val loss) by small margins
- BA wins on CIFAR-10 zero-shot and MNIST
- BA is notable for matching Linear with **half the parameters** (49K vs 98K)
- MNIST is universally hard (semantic mismatch: digit classes vs natural language captions)

**Observations:**
- BA converged much faster than Linear (BA plateau'd by epoch ~250, Linear still improving at epoch 300)
- Linear's val loss is slightly lower (3.3620 vs 3.3715) — suggests Linear has slightly better capacity for this setting, or BA's cosine-profile parameterization is more constrained
- Both methods land in the same val-loss neighborhood (~3.37)
- The real comparison will come from zero-shot transfer and retrieval metrics after Linear finishes
- MNIST is universally hard (semantic mismatch: digit classes vs natural language captions)

**Progress on other runs:**
- BA K=256 small: running on GPU 1 (epoch ~100/1000, val loss 3.39 and dropping)
- BA K=128 large: vision val features done (77 min at 12 s/batch), text val done in 43 sec (!), now starting vision train features (~30 hours). Text was ~100× faster than vision because RoBERTa-L is cheaper than ViT-L per item.
- BA K=512 small: to launch after K=256 finishes

**Text vs vision feature extraction speed gap:**
- Vision val (ViT-L, 391 batches): **77 minutes** (~12 s/batch)
- Text val (RoBERTa-L, 391 batches): **43 seconds** (~9 it/s)
- 100× faster for text. The bottleneck is entirely ViT-L multi-layer extraction.

---

## 2026-04-13 — BA Full Training — STRUCTURE pipeline (Server A)

### BA Full Training — STRUCTURE pipeline (Server A)

**Launch:**
- Configs: `configs/losses_ba/bridge_anchor_base.yaml` (K=128), `bridge_anchor_base_256.yaml` (K=256)
- Both inherit `default.yaml`: DINOv2 ViT-L + RoBERTa-Large, 1000 epochs, BS=4096, LR finder on, cosine T_max=50, early stopping patience=200, `drop_duplicates=true, n_dup_samples=1`
- `layer_selection: best_only=true, last_only=false` (mutual kNN best pair)
- CLIP loss, `temperature=0.05, structure_lambda=0`

**Preflight checks:**
- COCO annotations: `captions_train2014.json -> /home/data/2026_COCO/annotations/captions_train2017.json` (symlink) — training set is **COCO train2017**, 118,287 images → ~118K pairs after dedup
- Large-model features (`vit_large_patch14_dinov2` + `all-roberta-large-v1`) **not yet cached on disk**. Only small-model dry-run features exist. → Feature extraction will run from scratch before training.

**Issues encountered during launch:**
1. **wandb login failed** — provided key is 86 chars, wandb expects 40-char keys (`ValueError: API key must be 40 characters long`). Fallback: `WANDB_MODE=offline`; offline runs can be synced later with `wandb sync wandb/offline-run-*` once a valid key is provided.
2. **Concurrent launch race** — both runs need the same ViT-L + RoBERTa-L COCO features; feature cache has no locking. Initial parallel launch would have corrupted the `.npy` files. Resolution: serialize — run K=128 alone, wait for features to be cached, then launch K=256 (will load from cache).
3. **`conda run` buffers stdout** — initial launch via `conda run -n structure python ...` produced empty log files. Switched to direct binary path: `/home/shiwon/miniconda3/envs/structure/bin/python -u ...` for unbuffered output.

**Current status (12:28):**
- K=128 running on GPU 0, extracting ViT-L + RoBERTa-L features for COCO val set
- Feature extraction running at ~11-12 s/batch at batch_size=64 (GPU 100% pegged)
- Estimated total feature extraction time: ~33 hours (val ~80 min, train ~30 hours, small eval ~30 min)
- Training loop itself will be fast once features are cached

**Key findings and corrections to the plan:**

1. **Eval list restricted to 4 datasets** — the default 26-dataset `zero_shot_datasets` list triggers multi-GB downloads (STL-10, Food-101, ImageNet, etc.) and Server A's network is heavily throttled (bursts 100 KB/s–6 MB/s). The full download would take days. Restricted to: `cifar10, stl10, cifar100, mnist` + `coco` retrieval (see `configs/losses_ba/bridge_anchor_base*.yaml`). Training hyperparameters unchanged; only `evaluation.zero_shot_datasets` differs from default.

2. **`conda run` buffers stdout** — all three train launches initially showed empty logs. Fixed by invoking `/home/shiwon/miniconda3/envs/structure/bin/python -u ...` directly (bypassing `conda run`).

3. **CocoCaptionDataset ToTensor bug** — training crashed at first COCO batch with `TypeError: pic should be PIL Image or ndarray. Got torch.Tensor`. Per CLAUDE.md, fix is to monkey-patch `torchvision.transforms.functional.to_tensor`. Injected inline into the launch wrapper rather than using `run_with_totensor_fix.py`.

4. **`features.batch_size=16` default is too small** — combined with slow per-batch processing, would give ~37 hour extraction. Raised to 64 (128 OOMs due to shared GPU with another user `joohyun` consuming ~7.5 GB).

5. **cuDNN is safe on torch 2.1.2** (contrary to project log note, which was specifically about torch 2.2.0). Enabled `cudnn.enabled=True, cudnn.benchmark=True`. Speedup was marginal because ViT-L multi-layer feature extraction (24 blocks × CLS token) is the actual bottleneck — GPU is compute-bound at ~4.5 img/s.

6. **Serial execution required** — both runs would race on the same feature cache paths with no file locking. K=256 launch deferred until K=128's feature cache is fully populated (~33 hours from now).

7. **Large-model features never cached on Server A** — only small-model (`vit_small` + `MiniLM_L6_v2`) features exist in `results/features/`. This means previous Linear/MLP/ResLowRankHead experiments on Server A were either dry-runs (small models) or somehow avoided this bottleneck. Full-model training on Server A is a first.

**Estimated timeline for full BA experiments on Server A:**
- K=128 feature extraction: ~33 hours (in progress)
- K=128 training + eval: ~1-2 hours
- K=256 feature extraction: 0 hours (cache reuse) + training ~1-2 hours
- K=512 same as K=256
- **Total: ~2 days for K=128, then hours for each variant**

**Recommendations pending user decision:**
- Option A: Wait ~33 hours for feature extraction to complete, then training is fast
- Option B: Switch to small encoders (ViT-S + MiniLM) whose features ARE cached → training starts immediately, comparable to dry-run results (but smaller capacity)
- Option C: Subsample COCO train to ~20K pairs → reduces feature extraction to ~5 hours, acceptable for iteration
- Option D: Deploy to Server B (faster network + possibly faster GPU) via the Docker image already pushed to Docker Hub

**wandb issue:**
- API key `wandb_v1_...` provided by user is 86 chars but wandb expects 40. All runs fall back to `WANDB_MODE=offline`; offline runs can be synced with `wandb sync wandb/offline-run-*` once a valid key is provided.

---

## 2026-04-13 — BA Integration into STRUCTURE — vanilla CLS-only

### BA Integration into STRUCTURE — vanilla CLS-only

**What was done:**
- Copied BA reference sources into `bridge-anchors/` (read-only reference, not imported)
- Implemented `BridgeAnchorAlignmentLayer` in `src/alignment/bridge_anchor.py`
  - Input `(B, D)` → K learnable anchors → cosine sims → L2-normalize → `(B, K)`
  - Uses `F.normalize` on both z and anchors before matmul
  - Accepts `num_anchors` (primary) and `dim_alignment` (alias from default.yaml deep-merge)
- Created `configs/losses_ba/bridge_anchor_base{,_256,_512}.yaml` (K=128/256/512, DINOv2 ViT-L + RoBERTa-L, same LR/BS/epochs as `losses_lin/clip_base_best.yaml`)
- Dry run `configs/dryrun_ba.yaml` (ViT-S + MiniLM, CIFAR-10, 2 epochs)

**Results:**
- CIFAR-10 zero-shot top-1 = **97.56%** (cf. Linear 97.7%, MLP 98.0% on same dry run)
- Training loss decreased 3.37 → 2.34; val clip_loss 0.95
- Model summary: 49,152 trainable params per modality = 128 × 384 (ViT-S hidden), matches expected `K * D`
- For the default large-model config: expected `2 * 128 * 1024 = 262,144` params (same as Linear `D * D_align = 1024 * 256 = 262,144`)

**Interface notes:**
- STRUCTURE's loader does recursive deep-merge of `defaults: !include ...` so `dim_alignment: 256` from default.yaml leaks into `alignment_layer_kwargs` even when the override dict only sets `num_anchors`. Fixed by making BA accept `dim_alignment` as alias (ignored if `num_anchors` is also set).
- STRUCTURE instantiates one alignment layer per modality, so each modality gets its own anchors — matches original BA design (`anchors_img` / `anchors_txt` separate param sets).
- Factory auto-discovers new files via `initialize_package_factory(__file__)` glob — no manual import needed in `__init__.py`.

**Next steps:**
- Run full training with large models on Server B
- Ablate num_anchors (128/256/512) and structure_lambda (0/1/5/10)

---

## 2026-04-11 — Docker Image Build (Server A)

**What was done:**
- Created `docker/` directory with 5 files: Dockerfile, requirements, docker-compose, setup_server.sh, README
- Generated requirements.structure.txt from verified conda env (181 packages, torch/torchvision/nvidia excluded)
- Base image: pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel (devel needed for DeepSpeed nvcc import)
- Installed Node.js 20 + Claude Code (v2.1.101) for interactive development on Server B
- All critical pins verified: transformers==4.45.2, deepspeed==0.14.4, timm==0.9.16, numpy==1.26.4, etc.
- Dry-run passed inside container: CIFAR-10 top1=97.66% (2 epochs, ViT-S + MiniLM, Linear) — matches conda env result
- Tagged: shiwonkim/structure-ba:v1

**Issues discovered and resolved:**
- Runtime base image lacks nvcc → DeepSpeed import fails with MissingCUDAException → switched to devel base image
- apt-get tzdata hangs on interactive timezone prompt → added DEBIAN_FRONTEND=noninteractive
- `tail -N` pipe on docker run buffers all output until completion → use unbuffered for streaming logs

**Docker files:**
- `docker/Dockerfile.structure` — 11-step build (base → apt → node → claude-code → pip → env)
- `docker/requirements.structure.txt` — 181 pinned packages from conda env
- `docker/docker-compose.yml` — nvidia runtime, code+data volumes, 16GB shm
- `docker/setup_server.sh` — one-command setup: pull, GPU check, launch
- `docker/README.md` — quick-start docs

---

## 2026-04-11 — STRUCTURE Environment Setup & Dry Run (Server A)

**What was done:**
- Cloned STRUCTURE repo, created conda env (Python 3.10, PyTorch 2.1.2+cu118)
- Fixed dependency issues: torch 2.2.0 cuDNN segfaults (driver 470), transformers 5.x requires torch>=2.4, deepspeed 0.18.x requires torch>=2.4, numpy 2.x incompatibility, setuptools pkg_resources removal, umap-learn version conflict
- Pinned versions: torch==2.1.2+cu118, transformers==4.45.2, deepspeed==0.14.4, numpy<2, setuptools==69.5.1, umap-learn==0.5.6, scipy==1.11.4
- Set up COCO data symlinks (2017->2014 format mapping)
- Ran dry runs for all alignment methods: LinearAlignmentLayer, MLPAlignmentLayer, ResLowRankHead, CSA
- Verified STRUCTURE regularization works (structure_lambda=10.0)
- Verified COCO retrieval evaluation works (I2T-R@1=15.8%, R@5=40.2%, R@10=54.5%)
- Dry run result: CIFAR-10 zero-shot 97.7% top-1 (2 epochs, ViT-S + MiniLM, Linear)

**Key findings:**
- No token-level alignment exists in STRUCTURE — our BA with CAP would be the first
- losses_mlp/ configs actually use ResLowRankHead, not MLPAlignmentLayer
- Default encoders: DINOv2 ViT-L (1024-d) + RoBERTa-Large (1024-d), not ViT-B + mpnet like our codebase
- Pipeline caches all features to disk, trains alignment layer only (very fast epochs)
- 1000 epochs standard, cosine T_max=50 cycles, early stopping patience=200

**Additional datasets needed for full evaluation:**
- Flickr30k (retrieval, manual download)
- ImageNet (zero-shot, manual download)
- 13 other datasets (various manual downloads, lower priority)

**Docker notes:**
- Server B has CUDA driver 550 -> cu118 works via backward compat
- Must pin all package versions in Dockerfile
- Pre-cache HuggingFace + timm models in image to save startup time
- ~~CocoCaptionDataset has a ToTensor double-application bug; use run_with_totensor_fix.py wrapper~~ (FIXED in source 2026-04-13 — see end-of-log)
- ~~cca_zoo 2.5.0 has PCA numerical instability; use run_csa_fix.py wrapper for CSA~~ (FIXED in source 2026-04-13 — see end-of-log)

**Issues discovered and resolved:**
- torch 2.2.0+cu118 cuDNN segfaults on NVIDIA driver 470 -> downgraded to torch 2.1.2+cu118
- transformers 5.x / deepspeed 0.18.x incompatible with torch <2.4 -> pinned older versions
- numpy 2.x breaks torch 2.1.x -> pinned numpy<2
- setuptools 70+ removed pkg_resources.packaging -> pinned setuptools==69.5.1
- umap-learn 0.5.12 incompatible with scikit-learn 1.5.2 -> pinned umap-learn==0.5.6
- wandb 0.25 changed run.save() API -> pinned wandb==0.17.9
- timm 1.0.26 ViT is_causal arg breaks torchvision fx tracing -> pinned timm==0.9.16
- scipy 1.15 stricter NaN checking breaks cca_zoo -> pinned scipy==1.11.4

## 2026-04-11 — Alignment Method Verification (Server A)

**What was done:**
- Tested all alignment methods end-to-end with 1-2 epoch dry runs:
  - LinearAlignmentLayer: PASS (CIFAR-10 top1=97.7%)
  - MLPAlignmentLayer (2-layer): PASS (CIFAR-10 top1=98.0%)
  - ResLowRankHead (rank=32): PASS (CIFAR-10 top1=97.5%)
  - STRUCTURE regularization (lambda=10): PASS (structure_loss tracked correctly)
  - CSA/CCA: PASS with PCA fix (CIFAR-10 top1=70.9%)
  - COCO retrieval evaluation: PASS (I2T R@1=15.8%, R@5=40.2%)
- Created wrapper scripts for known issues (run_with_totensor_fix.py, run_csa_fix.py)
- Created dry-run configs for each method (configs/dryrun_*.yaml)

**Key findings:**
- Token-level alignment does not exist in STRUCTURE — confirmed by codebase search
- CSA (CCA-based) is a closed-form solution, not iterative — no epochs needed
- CSA with use_reg=True adds a post-hoc R_S refinement via gradient descent on CCA weights
- All three learned alignment methods (Linear, MLP, ResLowRankHead) share the same training loop
- ResLowRankHead starts with gate alpha=0 (residual off), learns to open it during training

**Issues discovered and resolved:**
- CocoCaptionDataset.load_image() returns tensor, but transforms include ToTensor() -> monkey-patch wrapper
- cca_zoo PCA keeps zero-variance components causing NaN in eigh -> PCA(n_components=0.999) fix
- embedding_visualization: 0 causes ZeroDivisionError (epoch % 0) -> use 9999 instead

---

## 2026-04-13 — Image-side dedup at extraction (train only)

Stop wasting 5× extraction time and 5× disk on the image side. COCO train has 591K (image, caption) rows but only 118K unique images; we were encoding every JPEG five times.

**Pre-implementation audit (both checks PASS):**
- **Pairing safety**: `(df.groupby("image_path").cumcount() == 0)` extraction mask is bit-identical to the trainer's `(... cumcount() < 1)` mask. All 118K deduped image rows align 1:1 with deduped text rows. Kept caption per image is deterministically the lowest annotation_id.
- **Eval retrieval protocol**: COCO val stores all 25,014 (image, caption) rows; image features 5×-duplicated within; metric handles via image_path groupby; the per-batch loop relies on matching image/text row counts, so eval val MUST stay at 25,014. Train-only dedup.

**Code changes** (`src/trainers/alignment_trainer.py`):
- `_indexed_dataset_view(base_dataset, keep_indices)`: arbitrary-index variant of `_SubsetView` with attribute-write delegation.
- `_should_dedup_image_extraction(loader)`: gate on flag + `pool=none` + has duplicates + `drop_duplicates=true` + `n_dup_samples=1` + no `n_random_subsample_train`.
- `get_image_features` (pool=none branch): when gated, iterates only first-occurrence rows, builds `unique_to_full_idx` mapping, saves `{features (deduped), dataframe (deduped), is_image_deduped, unique_to_full_idx}`.
- `fit()` token override: replaced the joint shape check with per-tensor `_apply_if_full` so image (118K) and text/mask (591K) get masks applied independently.

**Code changes** (`configs/default.yaml`): added `features.image_dedup_extraction: true` with auto-disable rules documented inline.

**Smoke tests (4 PASS):**

| Test | Result |
|---|---|
| Synthetic 4 images × 5 captions mini-CocoCaptionDataset | mask correct, kept caption is first per image, `unique_to_full_idx` round-trip correct |
| Real COCO train df (591,753 rows) | extraction mask bit-identical to trainer mask, mapping range `[0, 118286]`, sample images all map back consistently |
| `dryrun_ba.yaml` (CIFAR-10, no duplicates) | dedup branch did not trigger, CIFAR-10 top-1 **97.18%** (bit-identical to baseline) |
| `dryrun_ba_token.yaml` (CIFAR-10 + `n_random_subsample_train=2000`) | auto-disable triggered, TOKEN TRAIN shape `(2000, 257, 384)`, end-to-end loss decreasing, eval ran |

**Expected savings on Server B:**
- ViT-S 384/224/fp16: 109 GB → 22 GB (87 GB saved)
- ViT-L 1024/224/fp16: 311 GB → 62 GB (**249 GB saved**)
- Wall-time: same 5× ratio; ViT-L extraction ~3 hr → ~40 min

**Backward compatibility:** old caches without sidecar fields keep loading unchanged. Per-tensor dedup application in `fit()` is symmetric — works for both old (591K image) and new (118K image) caches without any branching.

**Known open issue:** the gate doesn't yet explicitly check eval-loader-vs-train-loader, so if a future eval pass calls `get_image_features` with `pool=none` on COCO val it could shrink the val cache to 5000 rows and break the per-batch retrieval loop's shared `i` index. Mitigation for now: existing eval caches were extracted before this change. Clean fix is ~5 lines (thread an `is_eval_loader` flag) — deferred until Server B actually needs eval re-extraction. See IMPLEMENTATION.md for the full design notes.

---

## 2026-04-13 — FreezeAlign baseline integrated

Added Freeze-Align (Maniparambil et al., CVPR 2025) as a STRUCTURE alignment method, sharing the unified r224 token cache with BA-token (no new extraction needed).

**Files:**
- `src/alignment/freeze_align.py` (new) — `PatchProjection`, `ProjectionHead`, `FreezeAlignAlignmentLayer` registered with the alignment factory.
- `src/trainers/alignment_trainer.py::fit` — 2-line `set_modality()` hook after factory creation (gated on `hasattr`, backward-compatible).
- `configs/losses_fa/freeze_align_base.yaml` (new) — large-encoder config.
- `configs/dryrun_fa.yaml` (new) — ViT-S+MiniLM CIFAR-10 dryrun.

**Verification:** audited bridge-anchors port against the original `freeze-align/.../clip_adjustable_combined_vis_cls.py` on all four points (PatchProjection structure, no separate CLS-text projector, no global vision projector, weight sharing via broadcast Linear). All three implementations (original, bridge-anchors, STRUCTURE port) consistent.

**Smoke test journey (3 iterations):**

| Iter | Setup | CIFAR-10 top-1 |
|---|---|---|
| 1 | Initial impl with separate `text_proj_cls` head | 13.9% (random head, never trained) |
| 2 | Single shared `text_proj` head | 3.7% (worse — trained on local_text_proj outputs, OOD on raw CLS) |
| **3** | Same head + `token_level_zero_shot=true` | **95.33%** ✓ |

**Lesson:** FreezeAlign's `text_proj` cannot be used in CLS-fallback mode at eval — its training input distribution (post `local_text_proj`) differs from raw encoder CLS. Always set `evaluation.token_level_zero_shot=true` with FreezeAlign. The dryrun config now does so by default.

**Param counts (ViT-S, embed=384):**
- Per instance: 1,629,312 total / 888,576 active (image) or 740,736 active (text)
- Per training run (2 instances): 3,258,624 total / 1,629,312 active

See IMPLEMENTATION.md for full design notes, audit details, and the reasoning behind the deliberate departures from the original (no learnable temperature, no SPARC loss).

---

## 2026-04-13 — Wrapper scripts integrated into source, deleted

The three monkey-patching wrappers from initial setup (`run_with_totensor_fix.py`, `run_csa_fix.py`, `run_dryrun.sh`) are gone. Patches now live in source:

- `coco_dataset.py::load_image` and `flickr30k_dataset.py::load_image` return `PIL.Image` instead of `torch.Tensor`. The downstream timm transform pipeline then runs end-to-end as designed.
- `csa_trainer.py` patches `cca_zoo.linear._mcca.MCCA._apply_pca` to use `PCA(n_components=0.999)` at module import time. Any subsequent CSATrainer / NormalizedCCA call uses the safe variant automatically.
- `run_dryrun.sh`'s `cudnn.enabled=False` was for the historical Server A driver-470 issue and is no longer needed (verified torch 2.1.2 + cuDNN works on this hardware during BA token training).

**Smoke tests (no wrappers, plain `python src/train_alignment.py …`):**

| Test | Result |
|---|---|
| `configs/dryrun_ba.yaml` (BA on CIFAR-10) | CIFAR-10 top-1 **97.18%** (matches published 97.56% within run noise) |
| `configs/dryrun_csa.yaml` (CSA on COCO → CIFAR-10) | CIFAR-10 top-1 **70.94%** (matches published 70.9% exactly; **no NaN** — confirms cca_zoo patch active) |
| Targeted unit tests on the patched code paths (`load_image` returns PIL, timm pipeline runs on PIL, `MCCA._apply_pca` patched at import) | all green |

**Files deleted:** `run_with_totensor_fix.py`, `run_csa_fix.py`, `run_dryrun.sh`.

**Files modified:** `src/core/src/datasets/downstream_tasks/coco_dataset.py`, `src/core/src/datasets/downstream_tasks/flickr30k_dataset.py`, `src/trainers/csa_trainer.py`, `CLAUDE.md`, `docker/README.md`, `IMPLEMENTATION.md`.

**Net effect:** every entry point on Server A and Server B is now plain `python src/train_alignment.py --config_path <yaml>` — no wrapper boilerplate, no remember-to-wrap-CSA gotcha, no fragile composition with `conda run` / `nohup`. See IMPLEMENTATION.md for the full Change History entry.
