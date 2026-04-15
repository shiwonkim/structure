# Experiment Directory

Tracking document for all experiment batches. Each entry records the date, data configuration, encoder setup, what was run, and where results are stored.

---

## Batch 1: Small Model Pilot (COCO 2017, ViT-S + MiniLM)

- **Date**: 2026-04-14
- **Data**: COCO 2017 train (118,287 images), NOT comparable to STRUCTURE paper (which uses COCO 2014 train 82K)
- **Encoders**: DINOv2 ViT-S/14 (384d) + all-MiniLM-L6-v2 (384d)
- **Resolution**: img_size=224
- **Layer selection**: img=L11, txt=L6 (mutual kNN)
- **Archive**: `archive/2026-04-14_coco2017_vits/`
  - `logs/` — training logs with final eval metrics
  - `checkpoints/` — best model weights
  - `features/` — extracted feature caches (ViT-S 224-res, 2017 split)
  - `wandb/` — offline wandb runs

### Runs completed (9):

| # | Method | Type | Config | Key result (COCO I2T R@1) |
|---|--------|------|--------|---------------------------|
| 1 | Linear d=512 | CLS | clip_base_small_d512.yaml | 27.5 |
| 2 | Linear+STR d=512 | CLS | clip_base_small_d512_struct.yaml | 27.7 |
| 3 | MLP d=512 | CLS | mlp_base_small_d512.yaml | 27.4 |
| 4 | MLP+STR d=512 | CLS | mlp_base_small_d512_struct.yaml | 27.8 |
| 5 | CLS BA K=128 | CLS | bridge_anchor_base_small_k128.yaml | 26.9 |
| 6 | CLS BA K=256 | CLS | bridge_anchor_base_small_k256.yaml | 27.6 |
| 7 | Token BA K=128 | Token | token_ba_small_k128.yaml | 41.4 |
| 8 | Token BA K=256 | Token | token_ba_small_k256.yaml | 44.5 |
| 9 | FreezeAlign | Token | freeze_align_small.yaml | 36.5 |

### Key findings:
- Token BA K=256 beats all methods on retrieval (+16.7pp over best CLS baseline)
- Token BA K=128 beats FreezeAlign at 33x fewer params
- CLS BA ≈ Linear/MLP — CAP is the key differentiator
- Results NOT directly comparable to STRUCTURE paper (different data split)

### Note:
These results used COCO 2017 train (118K images) instead of COCO 2014 train (82K). The STRUCTURE paper uses 2014. To enable direct comparison, Batch 2 onwards uses COCO 2014.

---

## Batch 2: Small Model Main Table (COCO 2014, ViT-S + MiniLM)

- **Date**: 2026-04-15
- **Data**: COCO 2014 train (82,783 images), matches STRUCTURE paper setup
- **Encoders**: DINOv2 ViT-S/14 (384d) + all-MiniLM-L6-v2 (384d)
- **Resolution**: img_size=224
- **Layer selection**: TBD (re-run at 224 on 2014 split)
- **Location**: `logs/`, `results/checkpoints/`, `results/features/`

### Planned runs (14, similar layer only):

| # | Method | Type | BS | dim_out | +STR |
|---|--------|------|----|---------|------|
| 1 | Token BA K=128 | Token | 4096 | 128 | No |
| 2 | Token BA K=256 | Token | 4096 | 256 | No |
| 3 | Token BA K=512 | Token | 4096 | 512 | No |
| 4 | CLS BA K=128 | CLS | 4096 | 128 | No |
| 5 | CLS BA K=256 | CLS | 4096 | 256 | No |
| 6 | CLS BA K=512 | CLS | 4096 | 512 | No |
| 7 | Linear d=512 | CLS | 4096 | 512 | No |
| 8 | Linear+STR d=512 | CLS | 4096 | 512 | Yes (λ=10) |
| 9 | MLP d=512 | CLS | 4096 | 512 | No |
| 10 | MLP+STR d=512 | CLS | 4096 | 512 | Yes (λ=10) |
| 11 | CSA | CLS | 4096 | 384 | No |
| 12 | CSA+STR | CLS | 4096 | 384 | Yes |
| 13 | FreezeAlign d=512 | Token | 4096 | 512 | No |
| 14 | FreezeAlign+STR d=512 | Token | 4096 | 512 | Yes (λ=10) |

### Status: PENDING — awaiting COCO 2014 image download + feature extraction

---

## Batch 3: Large Model Main Table (COCO 2014, ViT-L + RoBERTa)

- **Date**: TBD
- **Data**: COCO 2014 train (82,783 images)
- **Encoders**: DINOv2 ViT-L/14 (1024d) + all-roberta-large-v1 (1024d)
- **Resolution**: img_size=224
- **Layer selection**: TBD
- **Location**: TBD (likely Server B)

### Planned runs: same 14 methods as Batch 2, with encoder-appropriate dims

### Status: PENDING — after Batch 2 completes
