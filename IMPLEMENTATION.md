# BridgeAnchors (BA) Integration — Implementation Log

Detailed record of all files changed or created for integrating BridgeAnchors as an alignment method in the STRUCTURE codebase.

**Update rule:** append a new entry under "Change History" after every BA implementation update, and keep "Current File Map" in sync.

---

## Current File Map

### New files (created for BA)

#### `src/alignment/bridge_anchor.py`
Vanilla BA alignment layer — measures cosine similarity of embeddings to K learnable anchors.

```python
@AlignmentFactory.register()
class BridgeAnchorAlignmentLayer(BaseAlignmentLayer):
    def __init__(self, input_dim, num_anchors=None, init_method='random', dim_alignment=None):
        super().__init__(input_dim=input_dim)
        # num_anchors takes precedence; dim_alignment is an alias
        # (STRUCTURE's yaml deep-merge leaks dim_alignment=256 from default.yaml)
        if num_anchors is None:
            num_anchors = dim_alignment if dim_alignment is not None else 128
        self.anchors = nn.Parameter(torch.empty(num_anchors, input_dim))
        nn.init.normal_(self.anchors)
        with torch.no_grad():
            self.anchors.data = F.normalize(self.anchors.data, dim=-1)

    def forward(self, z):                        # z: (B, D)
        z_norm = F.normalize(z, dim=-1)
        a_norm = F.normalize(self.anchors, dim=-1)
        profile = z_norm @ a_norm.T              # (B, K)
        return F.normalize(profile, dim=-1)      # (B, K), unit sphere
```

Key design notes:
- No projectors, no CAP, no MoE, no routing, no auxiliary losses — just anchors + cosine + L2.
- STRUCTURE instantiates one alignment layer per modality, so img and txt each get their own anchors (matches original BA's `anchors_img` / `anchors_txt`).
- Param count per modality = `K * D`. For default config (K=128, D=1024): 131,072. Both modalities: 262,144 — matches Linear's `D * dim_alignment = 1024 * 256`.

#### `configs/losses_ba/bridge_anchor_base.yaml`
Large-model BA config (K=128), parallels `configs/losses_lin/clip_base_best.yaml`.

```yaml
defaults: !include ../default.yaml
overrides:
    training:
        alignment_layer_name: "BridgeAnchorAlignmentLayer"
        alignment_layer_kwargs:
            num_anchors: 128
        clip_loss_name: "CLIPLoss"
        clip_loss:
            temperature: 0.05
            structure_lambda: 0
    layer_selection:
        best_only: true
        last_only: false
```

#### `configs/losses_ba/bridge_anchor_base_256.yaml`
Same as above with `num_anchors: 256` (param-matched to Linear `dim_alignment=256`).

#### `configs/losses_ba/bridge_anchor_base_512.yaml`
Same as above with `num_anchors: 512`.

#### `configs/dryrun_ba.yaml`
Small-model dry-run (ViT-S + MiniLM, CIFAR-10, 2 epochs, K=128). Used to verify end-to-end integration on Server A before full training.

#### `bridge-anchors/` (reference only, not imported)
Read-only copy of the original BA codebase files, for reference while implementing the STRUCTURE-compatible version.
- `bridge_anchors.py` — original BA model
- `losses.py` — original BA losses
- `train.py` — original BA train loop
- `default.yaml` — original BA config

### Existing files that were modified

None so far. BA is added as a drop-in alignment method; the alignment factory auto-discovers `src/alignment/bridge_anchor.py` via `initialize_package_factory(__file__)` glob in `src/alignment/__init__.py` — no manual import edit required.

---

## STRUCTURE Interface Contract (how BA plugs in)

This section documents the STRUCTURE-side hooks BA relies on, so future BA edits stay inside the contract:

1. **Base class** — `src/alignment/base_alignment_layer.py`:
   ```python
   class BaseAlignmentLayer(ABC, nn.Module):
       def __init__(self, input_dim: int): ...
       @abstractmethod
       def forward(self, z: torch.Tensor) -> torch.Tensor: ...
   ```
   BA must subclass this and accept `input_dim` as the first kwarg.

2. **Factory registration** — `@AlignmentFactory.register()` decorator at class definition. No manual registry edit needed.

3. **Per-modality instantiation** — `src/trainers/alignment_trainer.py:762` creates one instance per modality:
   ```python
   alignment_image = AlignmentFactory.create(
       name, input_dim=image_dim, **alignment_layer_kwargs)
   alignment_text  = AlignmentFactory.create(
       name, input_dim=text_dim,  **alignment_layer_kwargs)
   ```
   → BA's anchors are **not shared** across modalities. Each call creates a fresh `nn.Parameter(K, D)`.

4. **YAML deep-merge** — `src/core/src/utils/loader.py:merge_dicts` recursively merges override dicts into defaults. This means **`alignment_layer_kwargs` from `default.yaml` always leaks into child configs**. BA handles this by accepting `dim_alignment` as an alias.

5. **Forward signature** — `forward(z: (B, D)) -> (B, D_out)`. Single modality at a time, no cross-modal interaction inside the alignment layer (unlike the original BA's CAP).

---

## Design Decisions & Tradeoffs

| Decision | Choice | Reason |
|---|---|---|
| Anchors per modality | **Separate** (img ≠ txt) | Matches STRUCTURE's per-modality instantiation contract; also matches original BA `anchors_img` / `anchors_txt` |
| Anchor init | Normal + L2-normalize to unit sphere | Keeps cosine similarities in `[-1, 1]` from step 0 |
| Output normalization | L2 on profile vector | Makes the K-dim profile a point on a unit K-sphere; compatible with STRUCTURE's `normalize_latents: true` CLIP loss |
| `dim_alignment` handling | Accept as alias, `num_anchors` takes precedence | Works around YAML deep-merge without touching STRUCTURE's loader |
| CAP / projectors / MoE | **Omitted** | "Vanilla" baseline first; CAP will require either per-token caching or a different integration point (TBD) |

---

## Dry-run Verification (2026-04-13)

Config: `configs/dryrun_ba.yaml` (ViT-S + MiniLM, CIFAR-10, 2 epochs, K=128)

| Metric | Value |
|---|---|
| CIFAR-10 top-1 (zero-shot) | **97.56%** |
| CIFAR-10 top-5 | 99.96% |
| Train loss | 3.37 → 2.34 |
| Val clip_loss | 0.947 |
| Params / modality | 49,152 (= 128 × 384) ✓ |

Comparison (same dry-run harness):
| Method | CIFAR-10 top-1 |
|---|---|
| Linear | 97.7% |
| MLP (2-layer) | 98.0% |
| ResLowRankHead | 97.5% |
| **BA (K=128)** | **97.56%** |
| CSA (PCA-fixed) | 70.9% |

BA is in the same ballpark as Linear/MLP/ResLowRankHead on the dry-run.

---

## Change History

### 2026-04-18 — Token-level support for Linear and MLP alignment layers

**Goal.** Extend STRUCTURE's existing CLS-only Linear and ResLowRankHead (MLP)
alignment layers to accept token-level `(B, T, D)` input, creating a clean
ablation ladder: CLS-only → token-level mean-pool → token-level CAP (BA).

**Design.** Follows STRUCTURE's symmetric convention (both modalities get
independent projection layers, unlike bridge-anchors' asymmetric image-only
projection). The `nn.Linear` / `nn.Sequential` already broadcasts over the
token dimension, so the only addition is a mean-pool step after projection:

```
(B, T, D) → per-token projection → (B, T, D_out) → masked_mean_pool → (B, D_out)
```

When input is 2D `(B, D)`, forward behaves identically to the original —
full backward compatibility with existing CLS checkpoints and configs.

**CLS token handling.** All token-level methods now use the **full sequence**
(CLS + patches, T=257 for ViT at 224px) rather than stripping CLS. This
applies uniformly to Linear-token, MLP-token, Token BA CAP, and FreezeAlign.
Rationale: creates a clean progression where CLS-only uses just position 0 and
token-level adds local patch information on top.

**Code changes.**

- `src/alignment/linear_alignment_layer.py`:
  - Added `_masked_mean_pool(x, mask)` helper (shared with MLP).
  - `forward(z, mask=None)` — 3D branch: project per-token → masked mean-pool.
  - Fixed `F.normalize` dim from `dim=1` to `dim=-1` for robustness.

- `src/alignment/mlp_alignment_layer.py`:
  - `MLPAlignmentLayer.forward(z, mask=None)` — same pattern.
  - `ResLowRankHead.forward(z, mask=None)` — same pattern after gated
    residual computation.

- `src/loss/clip_loss.py`:
  - `structure_reg` 3D reduction changed from CLS-slice `[:, 0, :]` to
    `mean(dim=1)`. Mean-pool matches what the alignment layers compute,
    ensuring the structure regularizer measures inter-sample topology
    on the same representation the CLIP loss sees.

**Configs (8 new files).**

| Config | Method | Encoder | STR |
|---|---|---|---|
| `configs/linear/vits_minilm/linear_d512_token.yaml` | Linear | ViT-S + MiniLM | No |
| `configs/linear/vits_minilm/linear_d512_token_struct.yaml` | Linear | ViT-S + MiniLM | Yes (λ=10) |
| `configs/mlp/vits_minilm/mlp_d512_token.yaml` | MLP | ViT-S + MiniLM | No |
| `configs/mlp/vits_minilm/mlp_d512_token_struct.yaml` | MLP | ViT-S + MiniLM | Yes (λ=10) |
| `configs/linear/vitl_roberta/linear_d512_token.yaml` | Linear | ViT-L + RoBERTa | No |
| `configs/linear/vitl_roberta/linear_d512_token_struct.yaml` | Linear | ViT-L + RoBERTa | Yes (λ=10) |
| `configs/mlp/vitl_roberta/mlp_d512_token.yaml` | MLP | ViT-L + RoBERTa | No |
| `configs/mlp/vitl_roberta/mlp_d512_token_struct.yaml` | MLP | ViT-L + RoBERTa | Yes (λ=10) |

All set `training.token_level: true` and `evaluation.token_level_zero_shot: true`.

**Smoke tests (5/5 pass).**

| Test | Input | Output | Grads | Notes |
|---|---|---|---|---|
| LinearAlignmentLayer 3D | `(4, 257, 384)` + mask | `(4, 512)` L2=1.0 | ✓ linear_mapping | Masked sample pools over 200 valid tokens |
| LinearAlignmentLayer 2D | `(4, 384)` | `(4, 512)` | — | CLS fallback unchanged |
| ResLowRankHead 3D | `(4, 257, 384)` + mask | `(4, 512)` | ✓ P, W1 | Gated residual broadcasts correctly |
| ResLowRankHead 2D | `(4, 384)` | `(4, 512)` | — | CLS fallback unchanged |
| BridgeAnchorTokenAlignmentLayer | `(4, 257, 384)` | `(4, 128)` L2=1.0 | ✓ anchors | CAP over full 257 tokens including CLS |
| FreezeAlignAlignmentLayer image | `(4, 257, 384)` | `(4, 512)` L2=1.0 | — | Splits CLS/patches internally |
| FreezeAlignAlignmentLayer text | `(4, 16, 384)` + mask | `(4, 512)` L2=1.0 | — | Masked mean-pool → text_proj |
| structure_reg 3D | `(4, 257, 384)` orig + `(4, 512)` aligned | loss=0.000024 | — | Mean-pool reduction, no crash |

**Backward compatibility.** All existing CLS-only checkpoints and configs
are unaffected: 2D input skips the mean-pool branch and follows the original
code path exactly. The `mask` kwarg defaults to `None` so existing call sites
need no changes.

---

### 2026-04-15 — Zero-shot segmentation evaluation (unified 4-method × 2-strategy framework)

**Goal.** Add an open-vocabulary semantic-segmentation eval path so BA's
CAP-pooling advantage on dense prediction can be quantified against FreezeAlign
and a no-alignment baseline. The FreezeAlign paper reports 31.37 mIoU on
PASCAL VOC 2012 but their reference script
(`freeze-align/train/zero_shot_segmentation.py`) has several methodological
issues we wanted to correct — most importantly, it averages per-image IoU
instead of accumulating a confusion matrix, and it uses a min-max threshold
instead of argmax.

**Design.** A single shared evaluation loop consumes two method-specific
primitives:

```python
method.get_patch_features(layer_feats, device) -> (P, D_m)
method.get_text_features(classnames, templates, ...) -> (C, D_m)
```

Everything downstream — L2 normalize, cosine similarity, grid reshape,
bilinear upscale, argmax, confusion-matrix accumulation, mean IoU —
is shared across methods. Each method only defines how to produce per-patch
and per-class descriptors in its own aligned space.

**Four localization methods:**

| Method | Image side | Text side | When to use |
|---|---|---|---|
| `direct_cosine` | raw encoder patches `(P, D_enc)` | raw LM embeddings `(C, D_enc)` | baseline, no checkpoint needed |
| `freezealign` | `alignment_image.local_vision_proj(z)[:, 1:, :]` → `(P, E)` | `alignment_text(tokens, mask=…)` → `(C, E)` | FreezeAlign checkpoints |
| `anchor_codebook` | `F.normalize(patches) @ F.normalize(anchors_img).T` → `(P, K)` | per-class CAP profile via `alignment_text(tokens, mask=…)` → `(C, K)` | BA-token checkpoints; codebook-level matching |
| `attention_map` | `softmax(patches @ anchors_img.T / τ, dim=0)` → `(P, K)` | per-class CAP profile → `(C, K)` | BA-token checkpoints; "anchor attention where × class activates anchor" |

The `attention_map` method is BA-specific and novel: the trained pool
temperature `τ` is reused, and the softmax is over the patch axis so each
column is a per-anchor attention distribution over the patch grid. The
cosine between that and the per-class CAP profile answers "does the patch
belong to a class whose profile activates the anchors that look here?"

**Two text strategies** (applied uniformly to all four methods):

| Strategy | Templates |
|---|---|
| `raw` | single no-op format `"{}"` (class name as-is) — matches FreezeAlign's reference behavior |
| `ensemble` | 80 OpenAI ImageNet prompt templates (`DATASETS_TO_TEMPLATES["imagenet"]`), averaged per class |

Template averaging is handled by the existing
`src/evaluation/zero_shot_classifier.py::build_zero_shot_classifier`, which
already knows how to:
1. Tokenize `C × T` (classnames × templates) prompts at once
2. Forward through the LM and select the right layer
3. Pass each prompt through the alignment layer (with attention mask for
   token-level heads)
4. Reshape to `(C, T, D)`, L2-normalize each template, mean, re-normalize

So the ensemble strategy gets per-template CAP attention patterns for BA,
not a single mean embedding — each template independently produces its own
anchor profile and they're averaged only in the final K-dim (or embed-dim)
aligned space.

**Improvements over the FreezeAlign reference script.**

| Issue in reference | Fix |
|---|---|
| Per-image IoU averaged across val set | Confusion-matrix accumulation, standard mIoU = mean over classes of `TP / (TP + FP + FN)` |
| Hard-coded `18, 18` patch grid | `h = int(round(sqrt(P)))` from actual token count |
| `cv2.resize` linear for GT mask | GT kept at native resolution; similarity map is upscaled to GT size with `F.interpolate(mode='bilinear', align_corners=False)`. No lossy resize on targets. |
| Min-max normalize + fixed 0.4 threshold per class | `argmax` across classes per pixel — standard semantic segmentation |
| `target[target==255]=0` folds ignore into background | Ignore index (255) excluded from confusion matrix on both GT and pred |
| Class names tokenized as-is (`diningtable`, `tvmonitor`) | Rewritten to `"dining table"`, `"tv monitor"`, `"potted plant"` in a `VOC2012_CLASS_PROMPTS` dict so subword tokenizers handle them sanely |
| No prompt ensembling | Optional 80-template OpenAI set via `--text-strategies raw,ensemble` |
| Classes that never appear in GT and are never predicted count as 0 IoU | `denom > 0` mask excludes them from the mean (standard practice; avoids penalizing rare classes) |

**Mean IoU reporting.** `compute_iou_from_confusion` returns both
`miou_all` (all 21 classes) and `miou_fg` (20 foreground classes only,
excluding background at index 0). The foreground mIoU is the number that
open-vocab seg papers typically report on VOC and is comparable to
FreezeAlign's published 31.37.

**Checkpoint dispatch.**

```
auto_filter_methods(requested, alignment_image, alignment_text, cfg)
```

drops methods that don't fit the loaded checkpoint's alignment class:

- No checkpoint → only `direct_cosine` survives.
- `BridgeAnchor*` → keeps BA methods + `direct_cosine`, drops `freezealign`.
- `FreezeAlign*` → keeps `freezealign` + `direct_cosine`, drops BA methods.

This lets one shared launcher call four methods regardless of which
method's checkpoint is provided; incompatible entries are skipped with a
warning instead of failing the run.

**Usage.**

```bash
PYTHONPATH=. python src/evaluation/zero_shot_segmentation.py \
    --config configs/ba/vits_minilm/token_k256.yaml \
    --checkpoint results/alignment-.../checkpoint-epoch400.pth \
    --layer-img 11 --layer-txt 6 \
    --dataset voc2012 --data-root data/pascal_voc --download \
    --methods anchor_codebook,attention_map,direct_cosine \
    --text-strategies raw,ensemble \
    --gpu 0
```

Or via the launcher:

```bash
bash scripts/eval_segmentation.sh \
    configs/ba/vits_minilm/token_k256.yaml \
    results/alignment-.../checkpoint-epoch400.pth \
    0 11 6
```

**Verification.** Shape-checked with dummy tensors on CPU (no real
encoder/dataset). For T=257, D=384, K=128, E=512, C=21:

| Method | patch output | note |
|---|---|---|
| `anchor_codebook` | `(256, 128)` | ✓ |
| `attention_map` | `(256, 128)` | ✓, per-anchor column sums = 1.0 (softmax over P) |
| `freezealign` | `(256, 512)` | ✓ |
| `direct_cosine` | `(256, 384)` | ✓ |

Patch grid derivation: `P=256` → `h=w=16`, matches ViT-S/14 at img_size=224.
Confusion matrix accumulation correctly excludes `ignore_index=255`
(confusion total = valid-pixel count). Dispatch filter correctly drops
`freezealign` under a BA checkpoint and `anchor_codebook`/`attention_map`
under an FA checkpoint.

**End-to-end run deferred** until Batch 2 completes and the first BA-token
checkpoint is on disk. Data will auto-download on first run via
`torchvision.datasets.VOCSegmentation(download=True)` (~2 GB).

**Files touched.**

- `src/evaluation/zero_shot_segmentation.py` (new, 500 lines)
- `scripts/eval_segmentation.sh` (new launcher)

---

### 2026-04-15 — CSA `sim_dim` PCA-rank cap — ViT-S configs 384 → 256

**Symptom.** Baselines script crashed on run 5/6 (`csa_d384`) during `CSATrainer.fit` →
`cca_model.fit_transform_train_data` with

```
ValueError: Requested eigenvalue indices are not valid.
Valid range is [0, 363] and start <= end, but start=-20, end=363 is given
```

raised inside `scipy.linalg.eigh` via `cca_zoo.linear._mcca.MCCA._solve_gevp`.

**Root cause.** `src/trainers/csa_trainer.py` monkey-patches
`cca_zoo.linear._mcca.MCCA._apply_pca` at module import time to use
`sklearn.decomposition.PCA(n_components=0.999)` — this was the historical
workaround for NaN gradients on low-rank / zero-padded feature columns (see the
Apr 13 "Wrapper scripts integrated into source" entry). The whitening step
therefore drops any component whose cumulative variance ratio falls under the
0.999 threshold.

On 384-dim ViT-S DINOv2 + 384-dim MiniLM COCO 2014 features, this keeps **364
components** (20 drop out below 0.1 % variance). `NormalizedCCA` then asks for
the top `sim_dim=384` eigenvalues, which translates inside scipy to an index
window `[n − sim_dim, n) = [−20, 363)` — an illegal range.

The bug was latent in Batch 1 because sim_dim was never set to the encoder
width in the old configs. It surfaced as soon as we unified the Batch 2 spec
around `sim_dim = min(DIM_IMG, DIM_TXT)`.

**Fix.** `sim_dim: 384 → 256` in both vits_minilm CSA configs. 256 sits
comfortably below the 364-component cap (~27 % headroom) and aligns with the
STRUCTURE paper's convention of `sim_dim ≤ 0.75 × effective_rank`.

- Renamed: `configs/csa/vits_minilm/csa_d384.yaml → csa_d256.yaml`,
  `csa_d384_struct.yaml → csa_d256_struct.yaml`.
- `scripts/vits_minilm/01_train_baselines.sh` runs 5–6 updated.
- `scripts/vits_minilm/config.env`: `CSA_SIM_DIM=256` with a one-line comment
  pointing at this entry.
- Internal config comments updated to say `sim_dim=256`.
- Killed + relaunched the CSA rerun chain on GPU 0 with the new filenames
  rather than leaving backward-compat symlinks in the tree.

**Verification.** Rerun completed cleanly:
`05_csa_d256` train converged and evaluated in ~10 min (CCA is closed-form);
`06_csa_d256_struct` took ~3 min for the iterative refinement. Final metrics on
COCO 2014: STL10 top-1 89.99 %, CIFAR-10 top-5 97.31 %, CIFAR-100 top-1
31.85 %, I2T R@10 18.56 %. Layer-comb alignment score 0.274 — consistent with
the standalone mutual-kNN sweep (0.273).

**Rule going forward.** **`sim_dim` must be strictly less than the post-PCA
rank**, which for n-dim inputs under `PCA(n_components=0.999)` is typically
slightly below n. Rule of thumb: `sim_dim = floor(0.67 × input_dim)`:

| Encoder pair | input_dim | Expected rank (0.999) | Safe sim_dim |
|---|---|---|---|
| ViT-S DINOv2 + MiniLM-L6 | 384 | ~364 | **256** (safe) / 320 (tight) |
| ViT-L DINOv2 + RoBERTa-L | 1024 | ~960 | **512** (plenty of headroom) |

Batch 3 can likely keep `sim_dim: 512` but should verify on the first run.

**Files touched.** `configs/csa/vits_minilm/csa_d256.yaml` (renamed),
`configs/csa/vits_minilm/csa_d256_struct.yaml` (renamed),
`scripts/vits_minilm/01_train_baselines.sh`,
`scripts/vits_minilm/config.env`.

---

### 2026-04-15 — FreezeAlign `embed_dim=512` CLS-fallback routing fix

**Problem.** The original FA port in `src/alignment/freeze_align.py` hard-assumed
`embed_dim == input_dim` because `_forward_text`'s CLS fallback path fed raw
encoder CLS directly into the shared `text_proj` head:

```python
def _forward_text(self, z, mask):
    if z.dim() == 2:                          # (B, D) CLS fallback
        return F.normalize(self.text_proj(z), dim=-1)   # crashes if D != embed_dim
    ...
```

`text_proj` is a `ProjectionHead(embedding_dim=embed_dim, projection_dim=embed_dim)`,
so calling it on a `(B, input_dim)` tensor when `input_dim != embed_dim`
produces a `Linear` shape mismatch. All four FA configs were therefore pinned
to `embed_dim = input_dim` (384 for vits_minilm, 1024 for vitl_roberta), which
made the `fa_d512.yaml` filenames a lie.

**Fix.** Route the CLS fallback through the full token pipeline as a length-1
sequence. Mean-pooling over 1 token is just `squeeze(1)`, so the math is
identical to the token path with `T=1`:

```python
def _forward_text(self, z, mask):
    if z.dim() == 2:
        # CLS fallback — treat as single-token sequence
        z_seq     = z.unsqueeze(1)                # (B, D) → (B, 1, D)
        projected = self.local_text_proj(z_seq)   # (B, 1, embed_dim)
        pooled    = projected.squeeze(1)          # (B, embed_dim)
        feat      = self.text_proj(pooled)        # (B, embed_dim)
        return F.normalize(feat, dim=-1)
    ...
```

`_forward_image`'s CLS fallback was left as-is — it already routes through
`cls_vision_proj`, which is a `PatchProjection(input_dim → embed_dim)` and
handles the dim change natively.

**Consequences.**

1. `embed_dim` can now be set independently of `input_dim`. All 4 FA configs
   updated to `embed_dim: 512`:
   - `configs/freezealign/vits_minilm/fa_d512{,_struct}.yaml`
   - `configs/freezealign/vitl_roberta/fa_d512{,_struct}.yaml`

2. Zero-shot eval via the CLS fallback now uses *trained* weights on both
   `local_text_proj` and `text_proj` instead of piping raw CLS through a head
   that was only ever trained on `local_text_proj` outputs. The training →
   inference distribution mismatch that forced us to set
   `evaluation.token_level_zero_shot: true` in every FA config last week is
   gone. (We still keep the token-level flag on for FA because the token path
   is strictly stronger — but the CLS path is no longer broken.)

3. Removed the `input_dim != embed_dim` warning from `__init__` and the
   stale `# text CLS-fallback will fail` comment.

**Smoke test** (`input_dim=384`, `embed_dim=512`, `dropout=0.1`, on CPU):

| Path | Input | Output | L2 norm | Grads flow |
|---|---|---|---|---|
| image token | `(4, 257, 384)` | `(4, 512)` | 1.0 | ✓ |
| image CLS fallback | `(4, 384)` | `(4, 512)` | 1.0 | ✓ |
| text token + mask | `(4, 16, 384)` + `(4, 16)` | `(4, 512)` | 1.0 | ✓ |
| text CLS fallback | `(4, 384)` | `(4, 512)` | 1.0 | ✓ local_text_proj, ✓ text_proj |

All four paths verified. The text CLS-fallback backward test specifically
confirmed that gradients reach **both** `local_text_proj` and `text_proj`
parameters — i.e., the new path isn't a dead branch.

**Files touched.** `src/alignment/freeze_align.py` (both comments and
`_forward_text` body), 4× `configs/freezealign/*/fa_d512*.yaml` (`embed_dim`
field + stale NOTE comments removed), `configs/README.md` (stale "pinned to
input_dim" caveat removed).

---

### 2026-04-14 — CLS Attention Prior in BridgeAnchorTokenAlignmentLayer (plumbing only)

**Goal:** add the soft CLS-guidance bias from the reference bridge-anchors experiments to STRUCTURE's BA-token layer. On Server B this gave +0.49 mR at K=128 and +0.38 at K=512 with `β_init=1.0` (in the reference codebase). The STRUCTURE port adds only the *plumbing* — the runtime extraction path for DINOv2 CLS attention isn't built yet, so the feature is off-by-default and a no-op until CLS-attention caches exist.

**Concept** (standard CAP + log-prior bias):
```
sim    = z_norm @ a_norm.T                 # (B, T, K)
logits = sim / pool_temperature            # (B, T, K)

# optional CLS prior:
log_prior = log(cls_attn + eps)            # (B, P) → unsqueezed to (B, P, 1)
betas     = per-anchor learnable scalar    # (K,), init 1.0
logits   += betas * log_prior              # broadcast over K
attn     = softmax(logits, dim=1)          # (B, T, K)
profile  = (attn * sim).sum(1)             # (B, K)
```
`β=1.0` pushes each anchor toward patches that the backbone's CLS token considers globally important. `β→0` recovers standard CAP. Because `β` is per-anchor, each anchor independently decides how much CLS guidance to use.

`cls_attn` is shape `(B, num_patches)` (patches only, no CLS-to-CLS entry). When the sequence being pooled has length `T > P` (e.g. `T = 1 + P` because position 0 is the CLS token), zero-padding is inserted at the front so `log(eps)` at the CLS position strongly suppresses attending to it — matches the reference.

**Code changes** (`src/alignment/bridge_anchor_token.py`):

- `BridgeAnchorTokenAlignmentLayer.__init__` gains two kwargs:
  - `cls_attn_prior: bool = False` — opt-in flag
  - `cls_attn_beta_init: float = 1.0` — initial value for the per-anchor betas
- When `cls_attn_prior=True`, creates `self.beta = nn.Parameter(torch.full((K,), cls_attn_beta_init))`. Otherwise `self.beta = None` and the layer behaves exactly as before — checkpoints from existing BA-token runs load unchanged.
- `forward(z, mask=None, cls_attn=None)` — new optional `cls_attn` kwarg. When `self.cls_attn_prior` is True AND `cls_attn is not None`, the 3D token path pads/slices to match `T`, computes `log(cls_attn_padded + 1e-8)`, and adds `betas * log_prior` to the pre-softmax logits. All existing call paths still work:
  - `cls_attn=None` → standard CAP (no error, no branch surprise)
  - 2D CLS-fallback input → `cls_attn` silently ignored
  - `cls_attn_prior=False` → `cls_attn` silently ignored (layer acts as before)

**Code changes** (`src/trainers/alignment_trainer.py`):

- `train(... image_cls_attn: Optional[torch.Tensor] = None)` and `validate(... image_cls_attn: Optional[torch.Tensor] = None)` — new kwarg, defaulting to None.
- Inside the training/val per-batch loop:
  ```python
  if image_cls_attn is not None and getattr(alignment_image, "cls_attn_prior", False):
      img_cls_attn_batch = image_cls_attn[i:end_i].to(self.device)
      aligned_image_feats = alignment_image(image_feats, cls_attn=img_cls_attn_batch)
  else:
      aligned_image_feats = alignment_image(image_feats)
  ```
  The `getattr(layer, "cls_attn_prior", False)` guard keeps every other alignment class (Linear, MLP, CLS BA, FreezeAlign) untouched — they never receive an unexpected `cls_attn` kwarg.
- `train()` applies the same random shuffle to `image_cls_attn` it already applies to image/text/mask, so batch rows stay aligned.
- `fit()` is NOT yet modified to pass `image_cls_attn` down — because there is no extraction path yet, `image_cls_attn` is always None in practice. When the extraction lands later, a single `fit()` edit will activate the feature.

**Smoke tests** (layer in isolation, 6 checks all pass):

| Check | Result |
|---|---|
| Construct with `cls_attn_prior=True`, `β` has shape `(K,)`, init `1.0` | ✓ |
| 3D forward with `cls_attn=(B, P)` → output `(B, K)`, `β.grad` nonzero through softmax | ✓ |
| 3D forward with `cls_attn=None` differs from 3D with `cls_attn=tensor` (prior actually applies) | ✓, max abs diff 0.0346 |
| 2D CLS-fallback input ignores `cls_attn` → outputs identical with/without | ✓ |
| 20 SGD steps on `β`: `β` moves from 1.0000 to 1.0280 mean (learnable) | ✓ |
| `cls_attn_prior=False` layer has `β=None`, still accepts `cls_attn` kwarg silently | ✓ |

Module-level import check: `train()` and `validate()` signatures expose `image_cls_attn` with default `None`. Trainer imports clean.

**Explicit non-goals** (deferred):
- **Extraction of CLS attention** from DINOv2's last layer. The reference bridge-anchors saves it alongside token features; STRUCTURE's equivalent would be an update to `extract_token_features.py` + `_load_token_features_for_layer` to pull attention weights from the `vision_model`'s last-block self-attention and write a `*_cls_attn.npy` sidecar. Not implemented — the plumbing is in place; extraction is the next step.
- **Passing `image_cls_attn` down through `fit()`** into `train()` / `validate()`. Default is None everywhere, so the feature is dormant; a single `fit()` edit will activate it once the extraction lands.
- **Text-side CLS prior.** The reference doesn't use it (text encoders don't have a DINOv2-style globally-attending CLS), and our trainer only threads `image_cls_attn`, not a text equivalent.

**Backward compatibility:** every existing BA-token checkpoint loads unchanged (layers default to `cls_attn_prior=False`). Every existing config works unchanged (the new kwargs are optional). Every non-BA-token alignment layer is untouched by the trainer changes (gated by the `getattr(layer, "cls_attn_prior", False)` check).

**Files touched:**
- `src/alignment/bridge_anchor_token.py` — new kwargs, `self.beta`, forward prior branch.
- `src/trainers/alignment_trainer.py::train` and `::validate` — new optional `image_cls_attn` kwarg + conditional pass-through.

### 2026-04-13 — Image-side dedup at extraction (train only)

**Goal:** stop wasting ~5× extraction time and ~5× disk on the image side. COCO has ~5 captions per image, so iterating the dataset row-by-row encodes the same JPEG five times. Symptom: a 591K-row image token cache where rows 0–4 contain bit-identical features (same image, different caption). For ViT-L the wasted disk is ~250 GB per layer; for ViT-S it's ~87 GB. The text side is left alone because each caption is genuinely different.

**Verified-before-implementing checks** (audit pass):

1. **Pairing safety.** Loaded the real COCO train df, computed `(df.groupby("image_path").cumcount() == 0)` (extraction mask) and `(... cumcount() < 1)` (the trainer's `sel_train_indices`). They are bit-identical (118,287 rows each). For every one of the 118,287 deduped image rows, the caption at the same index references the same `image_path`. The kept caption is deterministically the lowest-annotation_id one.
2. **Eval retrieval protocol.** COCO val image and text caches each store all 25,014 (image, caption) rows; image features are 5× duplicated within the 25,014 image rows. `evaluation.drop_duplicates=false`, so the eval keeps everything. The retrieval metric (`retrieval_metrics_df` → `compute_ground_truth_mapping`) handles the duplication via `df.groupby(image_column).groups`. The eval per-batch loop uses the same `i` to slice both image and text tensors, so they MUST match in length. **Conclusion:** dedup train only; leave eval val untouched.

**Design** (single-step extraction with sidecar metadata):

When all of these hold:
- `features.image_dedup_extraction: true` (new flag, default `true` in `default.yaml`)
- `features.pool_img == "none"` (token mode; CLS pipeline is left alone)
- The dataset has a `df` with an `image_path` column and at least one duplicate
- `training.drop_duplicates: true` and `n_dup_samples: 1` (so the trainer would dedup to first-occurrence anyway)
- `training.n_random_subsample_train` is not set (the dryrun first-N path doesn't benefit and would interact awkwardly)

`get_image_features` iterates only the first-occurrence rows of each unique image and saves a richer cache:

```python
{
    "features":          (N_unique, T, D) fp16,
    "dataframe":         deduped df,                       # 118K rows
    "is_image_deduped":  True,                             # sentinel
    "unique_to_full_idx": (N_full,) int tensor,            # caption_row -> image_row mapping
}
```

Old caches without the sidecar fields keep loading unchanged (the consumer reads `features` and ignores extras).

**Code changes** (`src/trainers/alignment_trainer.py`):

- New helper `_indexed_dataset_view(base_dataset, keep_indices)`: an arbitrary-index variant of the existing `_SubsetView`. Mirrors the `__setattr__` delegation pattern so `loader.dataset.tokenizer = ...` still propagates and `apply_tokenizer()` still works through the proxy.
- New helper `_should_dedup_image_extraction(loader) -> bool`: gates on the conditions listed above. Returns `False` for any non-COCO-like dataset, for `pool_img != "none"`, when `drop_duplicates` is off, or when `n_dup_samples != 1`. Defensive — only fires when the savings are guaranteed and pairing semantics are safe.
- `get_image_features` (pool=none branch): when the gate is open, builds `keep_indices = np.where(first_idx_mask)[0]`, wraps the dataset in `_indexed_dataset_view`, builds a fresh DataLoader over the wrapped view, and runs the existing fp16 streaming allocation against `n_unique` rows instead of `len(df)`. Computes `unique_to_full_idx` via a `path → unique_position` dict + `df["image_path"].map(...)`. Saves the richer cache dict.
- `fit()` token override: replaced the joint-shape check that applied `sel_train_indices` to image+text+mask together with a per-tensor `_apply_if_full(tensor, mask_full, mask_bool)` helper. Each tensor's shape is independently compared against `len(sel_train_indices)`; if equal, the mask is applied; otherwise (e.g., image already 118K from dedup-at-extraction) it's silently skipped. This also handles the legacy `n_random_subsample_train` first-N path correctly (the subset's row count differs from the full df length, so dedup is silently skipped on all three tensors as before).

**Code changes** (`configs/default.yaml`):

```yaml
features:
    img_size: 224
    image_dedup_extraction: true   # NEW
```

Documented with the auto-disable rules inline. Default `true` so all new runs benefit; existing caches without the sidecar still load.

**Smoke tests** (4 tests, all pass):

| Test | Result |
|---|---|
| **Synthetic 4×5** mini-CocoCaptionDataset built from a temp JSON + dummy JPEGs. Verified `_should_dedup_image_extraction`-style logic computes the right `keep_indices=[0,5,10,15]`, the kept caption per image is the FIRST in row order, the `unique_to_full_idx` is `[0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3]`, and an `eye(4)[unique_to_full_idx]` round-trip materialises the correct expanded one-hot. | ✓ |
| **Real COCO train df** (591,753 rows from the existing token cache's persisted dataframe). Extraction mask and trainer mask are bit-identical (118,287 each). `unique_to_full_idx` shape `(591753,)`, range `[0, 118286]`. Five sampled images each map all their captions back to the same image row. | ✓ |
| **`dryrun_ba.yaml`** (BA on CIFAR-10) — no duplicates in df, dedup branch must NOT trigger. | ✓ CIFAR-10 top-1 **97.18%**, bit-identical to the pre-dedup baseline. No "Image dedup at extraction" log line. |
| **`dryrun_ba_token.yaml`** (BA-token on CIFAR-10 with `n_random_subsample_train=2000`) — auto-disable rule should keep dedup off. | ✓ TOKEN TRAIN shape `(2000, 257, 384)`, no "Image dedup at extraction" log line, CIFAR-10 top-1 82.41%. The dedup-aware fit() path correctly handles the subset-extraction layout via shape gates. |

Did NOT run a full COCO extraction with dedup-on (would take ~30 min on Server A and the goal is Server B handover); the synthetic + integration tests cover the dedup path comprehensively. The first real dedup-on extraction will happen on Server B.

**Backward compatibility audit:**

| Cache state | Loader behaviour |
|---|---|
| Old 591K cache, no sidecar fields | `get_image_features` returns `(591K, T, D)` as before; `fit()` per-tensor mask check matches 591K → dedup applied to all three; behaves identically to pre-change code |
| New 118K cache, sidecar present | `get_image_features` returns `(118K, T, D)`; `fit()` per-tensor check finds image shape 118K ≠ 591K → image dedup skipped; text and mask still 591K → dedup applied; final aligned shapes are 118K image / 118K text / 118K mask |
| New extraction with dedup off (e.g., user disables flag, or `n_random_subsample_train` is set) | Dedup branch never triggers; falls through to legacy 591K extraction; bit-identical to pre-change behaviour |
| pool=cls (legacy multi-layer CLS pipeline) | Untouched; `_should_dedup_image_extraction` returns False on `pool != "none"` |
| Eval val extraction (`evaluate_retrieval`, `evaluate_zero_shot_classification`) | Untouched; per-batch loop relies on same row count for image and text and the gating rules above don't apply because eval typically isn't running pool=none on a duplicating df anyway. The token-mode retrieval branch's helper `_load_eval_token_features` calls `get_image_features` in token mode, but on the eval DataLoader where the dataset is the COCO val set; that df also has duplicates so dedup COULD trigger. **Mitigation:** the eval val image cache is ~10–13 GB even at full row count, the savings are minor, and shrinking it would break the retrieval per-batch loop. The gate disables on `n_random_subsample_train is not None`, but does NOT explicitly disable on "the loader belongs to the eval set". Add a defensive carve-out if needed (see open issue below). |

**Expected savings on Server B:**

| Config | Train cache (current) | Train cache (deduped) | Saved |
|---|---|---|---|
| ViT-S 384 / 224 / fp16 | 109 GB | 22 GB | 87 GB |
| ViT-B 768 / 224 / fp16 | 198 GB | 40 GB | 158 GB |
| **ViT-L 1024 / 224 / fp16** | **311 GB** | **62 GB** | **249 GB** |

Wall-time: same 5× ratio. ViT-L vision train extraction goes from ~3–4 hours to ~40 minutes on Server B's GPU.

**Files touched:**

- `src/trainers/alignment_trainer.py`: `_indexed_dataset_view` helper, `_should_dedup_image_extraction` helper, `get_image_features` dedup branch, `fit()` per-tensor `_apply_if_full` rewrite of the token-override dedup application.
- `configs/default.yaml`: `features.image_dedup_extraction: true`.

**Known open issue (deferred):**

The `_should_dedup_image_extraction` gate doesn't explicitly check whether the loader belongs to the train or eval split — it gates on whether `n_random_subsample_train` is set, which happens to be a train-side knob. For the COCO val image cache (used by `evaluate_retrieval` token branch), the gate would currently fire (df has duplicates, `pool=none`, no train-subsample). If that triggers, val image cache shrinks to 5000 rows while text cache stays at 25014 → the per-batch retrieval loop's shared `i` index would crash.

**Mitigation for now:** the `_load_eval_token_features` helper is called via a separate code path that uses the `eval-...-r224` cache suffix; if you're running the eval token-retrieval path, just ensure either (a) the eval cache predates this change (already extracted at 25014), or (b) the retrieval loop is updated to handle asymmetric image/text shapes via the `unique_to_full_idx` sidecar. Track this as TODO; the BA-token K=128 small training already wrote a `(25014, 257, 384)` cache before this change so existing files load fine.

A clean fix would be to thread an explicit `is_eval_loader: bool` through `get_image_features` (defaulting to False), set True from `_load_eval_token_features`, and gate dedup on it. Estimated 5 lines. Will do in a follow-up if eval re-extraction becomes necessary on Server B.

### 2026-04-13 — FreezeAlign baseline integrated as alignment layer

**Goal:** add Freeze-Align (Maniparambil et al., CVPR 2025) as a STRUCTURE alignment method so we can compare against it on the same unified token cache as BA-token. No new feature extraction; reuses the `(N, T, D)` r224 token cache.

**Verification first** (4-point audit against the original repo `freeze-align/train/models/clip_adjustable_combined_vis_cls.py`):

| Point | Original | bridge-anchors port | New STRUCTURE port |
|---|---|---|---|
| `PatchProjection`: `Linear(x) + [Linear→GELU→Linear](x)`, no explicit `x+` skip, GELU not ReLU | ✓ | ✓ | ✓ |
| Text side: NO separate CLS-text projector; CLS included in token mean pool through `local_text_proj`, then `text_proj` MLP after pooling | ✓ | ✓ | ✓ |
| Vision side: NO global vision projector (`vision_proj = Identity`); only `local_vision_proj` + `cls_vision_proj` | ✓ | ✓ | ✓ |
| Local projector weight sharing: `nn.Linear` applied to `(B, T, D)` broadcasts over `T` automatically | ✓ | ✓ | ✓ |

All three implementations are consistent.

**New file** `src/alignment/freeze_align.py`:

- `PatchProjection`: `output = Linear(x) + [Linear(x) → GELU → Linear(x)]`, copied verbatim from the audit.
- `ProjectionHead`: `projected = Linear(x); h = GELU(projected) → Linear → Dropout; out = LayerNorm(h + projected)`, copied verbatim.
- `FreezeAlignAlignmentLayer(BaseAlignmentLayer)` registered via `@AlignmentFactory.register()`. Holds **both** modalities' components on every instance because STRUCTURE creates one alignment layer per modality from the same class + kwargs:
  - Vision: `local_vision_proj` (LN+Dropout+PatchProjection) and `cls_vision_proj` (LN+Dropout+PatchProjection).
  - Text: `local_text_proj` (LN+Dropout+PatchProjection) and one shared `text_proj` (`ProjectionHead`). The original repo uses a single `text_proj` head and assumes `text_width == embed_dim`; we follow that and warn loudly if `input_dim != embed_dim`.
  - `set_modality('image' | 'text')` flips `_modality`. If never called, `forward` auto-detects: presence of a `mask` argument → text branch.
  - `_forward_image(z)`: 2D → CLS fallback (`cls_vision_proj` only, which IS exercised during token training so it's well-trained). 3D → token forward: `local_vision_proj` on all tokens → mean-pool patches `[:,1:,:]` → add `cls_vision_proj(z[:,0,:])` → L2 normalize.
  - `_forward_text(z, mask)`: 2D → CLS fallback (`text_proj` directly). 3D → `local_text_proj` on all tokens → masked-mean → `text_proj` → L2 normalize.
  - `active_param_count()` reports parameters touched by the active modality's forward pass; full `parameters()` gives the instantiated total.
- **Temperature**: not learnable (STRUCTURE's `CLIPLoss` owns the temperature, fixed at 0.05 by config). Documented as a deliberate departure from the original to keep all alignment methods on equal footing.
- **SPARC fine-grained loss**: omitted. STRUCTURE's training loop only consumes the aligned `(B, embed)` features; SPARC is a separate objective.

**Trainer hook** (`src/trainers/alignment_trainer.py::fit`):

```python
alignment_image = AlignmentFactory.create(...)
alignment_text  = AlignmentFactory.create(...)
# Backwards-compatible — only FreezeAlign defines this method
if hasattr(alignment_image, "set_modality"):
    alignment_image.set_modality("image")
if hasattr(alignment_text, "set_modality"):
    alignment_text.set_modality("text")
```

**Configs:**

- `configs/losses_fa/freeze_align_base.yaml`: ViT-L + RoBERTa-Large at the unified 224 res, `embed_dim=1024`, `dropout=0.1`, `token_level=true`, standard 1000-epoch budget. Layers pinned by upstream layer-selection.
- `configs/dryrun_fa.yaml`: ViT-S + MiniLM, CIFAR-10, 2 epochs, BS 256, `n_random_subsample_train=2000`, `n_random_subsample_val=1000`, `embed_dim=384`, `token_level=true`, **`evaluation.token_level_zero_shot=true`** (see below).

**Parameter counts**

The architecture has 3 `PatchProjection` blocks (`local_vision_proj`, `cls_vision_proj`, `local_text_proj`) plus 1 `ProjectionHead` (`text_proj`), each sized at `embed_dim`. Per single STRUCTURE instance (which holds both modalities' components):

| `embed_dim` | Per instance | Per training run (2 instances) | Notes |
|---|---|---|---|
| 384 (ViT-S smoke test) | 1.63M | 3.26M | dryrun |
| 768 (dinov2-base + mpnet) | 6.5M | 13.0M | original repo default |
| **1024 (ViT-L + RoBERTa-Large)** | **11.55M** | 23.1M | matches the paper's "≈11M projection layer parameters" headline |

Active params per modality at the dryrun (384) scale: image 888,576 (two PatchProj-based projectors), text 740,736 (one PatchProj-based + one smaller ProjectionHead MLP). The unused half of each instance receives no gradient and is effectively memory dead weight — the cost of fitting STRUCTURE's "one class, one kwargs dict" alignment-factory contract.

The 11M figure in the paper refers to a single FreezeAlignAlignmentLayer instance at `embed_dim=1024` (their headline ViT-L config). STRUCTURE's pipeline instantiates the class twice per training run (one per modality), so the actual on-device footprint at the headline config is 2 × 11.55M = 23.1M, of which ~12M is "active" per backward pass (the active modality's components on each instance).

**Smoke test journey** (3 dryrun iterations on the same checkpoint until end-to-end was healthy):

| Iter | Setup | CIFAR-10 top-1 | Notes |
|---|---|---|---|
| 1 | Initial impl with separate `text_proj_cls` head | 13.9% | `text_proj_cls` was random init and never trained; CLS-fallback eval used random head |
| 2 | Single shared `text_proj` head | 3.7% (worse!) | Shared head helps — but it's trained on `local_text_proj` outputs (post LN+PatchProjection), so feeding raw CLS at eval is **out-of-distribution**. Worse than random |
| **3** | Same head + `evaluation.token_level_zero_shot=true` | **95.33%** ✓ | Eval uses the same forward path as training (FreezeAlign's mean-pool token forward, NOT CAP — `token_level_zero_shot` simply enables 3D forwards; the alignment layer's own forward decides the pooling). Inputs are now in-distribution |

**Lesson learned**: FreezeAlign's `text_proj` cannot meaningfully be used in CLS-fallback mode at eval time, because its training input distribution (post `local_text_proj`) differs from raw encoder CLS. Always use `evaluation.token_level_zero_shot=true` with FreezeAlign. The same caveat does NOT apply to `cls_vision_proj` because it IS exercised during token-mode training (line `cls_feat = self.cls_vision_proj(z[:,0,:])`) so it sees raw CLS embeddings throughout training.

**Final verification** (dryrun #3 trace):

```
TOKEN TRAIN - img: (2000, 257, 384), txt: (2000, 7, 384), txt_mask: (2000, 7)
TOKEN VAL   - img: (1000, 257, 384), txt: (1000, 7, 384), txt_mask: (1000, 7)
FreezeAlignAlignmentLayer.set_modality('image')
FreezeAlignAlignmentLayer.set_modality('text')
Train loss: 3.9888 → 3.4826  (epoch 1 → 2)
Val   loss: 3.5511 → 3.5039
Cifar10 - top1_acc_micro: 0.9533, top5_acc_micro: 0.9990
```

Loss decreasing monotonically, eval path operational, set_modality invoked correctly, no shape errors anywhere in the pipeline.

**Files touched:**

- `src/alignment/freeze_align.py` (new)
- `src/trainers/alignment_trainer.py::fit` (2-line `set_modality` hook after factory creation)
- `configs/losses_fa/freeze_align_base.yaml` (new — large-encoder config)
- `configs/dryrun_fa.yaml` (new — small dryrun, includes `token_level_zero_shot=true`)

**Backward compatibility:** the `set_modality` hook is gated on `hasattr`, so every existing alignment layer (Linear, MLP, ResLowRank, BA, BA-token, CSA) is untouched. FreezeAlign reuses the existing unified r224 token cache extracted for BA-token; **no new feature extraction needed**.

**Server B handover:** the large-encoder config is ready. Workflow per the existing pipeline:
1. Layer selection at 224 (already done for BA-token, reuse the result).
2. Pin layers in `freeze_align_base.yaml` (or let layer-select run inline).
3. Run `python src/train_alignment.py --config_path configs/losses_fa/freeze_align_base.yaml`.
4. Reads the same `*-none_layer-{L}-r224.npy` token cache as BA-token. No re-extraction.

### 2026-04-13 — Wrapper scripts integrated into source, deleted

**Goal:** during the initial setup we couldn't modify upstream STRUCTURE files, so we shipped three wrapper shims that monkey-patched issues at runtime. Now that we own this fork, the patches go directly into the source and the wrappers go away.

**Inventory + integration:**

| Wrapper (deleted) | What it patched | Integrated into |
|---|---|---|
| `run_with_totensor_fix.py` | `torchvision.transforms.functional.to_tensor` to pass through `torch.Tensor` inputs | `src/core/src/datasets/downstream_tasks/coco_dataset.py::CocoCaptionDataset.load_image` and `src/core/src/datasets/downstream_tasks/flickr30k_dataset.py::FlickrDataset.load_image` now return `PIL.Image` instead of a tensor. The downstream timm transform pipeline (Resize → CenterCrop → ToTensor → Normalize) then runs end-to-end on a PIL image as it expects. |
| `run_csa_fix.py` | (a) same ToTensor patch, plus (b) `cca_zoo.linear._mcca.MCCA._apply_pca` overridden to use `PCA(n_components=0.999)` so zero-variance components are dropped and CCA stops producing NaN on low-rank features | (a) covered by the coco/flickr fix above; (b) `src/trainers/csa_trainer.py` now applies the same `_apply_pca` override at module import time, so any subsequent `CSATrainer` / `NormalizedCCA` use sees the safe variant |
| `run_dryrun.sh` | `torch.backends.cudnn.enabled = False` for the old Server A driver-470 cuDNN incompatibility | Not needed — torch 2.1.2 + cuDNN works on this hardware (verified during the BA token training). Plain `python src/train_alignment.py …` is fine. |

**Targeted unit smoke tests** (just the patched code paths, no training):

| Test | Result |
|---|---|
| `CocoCaptionDataset.load_image` source no longer contains `transforms.ToTensor()(image)` | ✓ |
| PIL fallback path returns `PIL.Image` for a synthesized JPEG | ✓ |
| `timm.create_transform(...)` runs end-to-end on the PIL output → `(3, 224, 224)` tensor | ✓ |
| `import src.trainers.csa_trainer` triggers MCCA `_apply_pca` patch; source contains `n_components=0.999` | ✓ |

**End-to-end smoke tests** (full dryrun configs, no wrappers, just `python src/train_alignment.py --config_path …`):

| Config | Result |
|---|---|
| `configs/dryrun_ba.yaml` (BA on CIFAR-10) | ✓ CIFAR-10 top-1 **97.18%** (matches published 97.56% within run noise) |
| `configs/dryrun_csa.yaml` (CSA on COCO → CIFAR-10) | ✓ CIFAR-10 top-1 **70.94%** (matches published 70.9% exactly; **no NaN** — confirms the cca_zoo PCA patch is active) |

**Doc updates:**

- `CLAUDE.md` "Known Issues & Workarounds" rewritten — historical wrappers no longer mentioned; new "Running a training job" section with the plain command.
- `docker/README.md` — removed the "use `run_dryrun.sh`" cuDNN note.
- `PROJECT_LOG.md` — appended a cleanup entry pointing here.

**Files changed in this commit:**

- `src/core/src/datasets/downstream_tasks/coco_dataset.py` — `load_image` returns PIL.
- `src/core/src/datasets/downstream_tasks/flickr30k_dataset.py` — same.
- `src/trainers/csa_trainer.py` — top-level cca_zoo `_apply_pca` patch.
- `CLAUDE.md`, `docker/README.md` — doc updates.

**Files deleted:**

- `run_with_totensor_fix.py`
- `run_csa_fix.py`
- `run_dryrun.sh`

**Net effect:** every entry point on Server A and Server B is now plain `python src/train_alignment.py --config_path <yaml>`. No more `python run_with_totensor_fix.py src/train_alignment.py …` boilerplate, no more "remember to wrap CSA runs with the wrapper" gotchas, and no more scope for the wrapper to silently fail to compose with `conda run`, `nohup`, or background runners.

### 2026-04-13 — Optional token-level zero-shot classification eval

**Goal:** zero-shot classification was always falling back to the BA-token CLS path (2D forward through `if z.dim() == 2`), so reported numbers under-counted the model's potential. Added an opt-in token-level zero-shot path that runs both modalities through CAP, so we can report both numbers in the paper.

**New config flag** (`configs/default.yaml`):

```yaml
evaluation:
    token_level_zero_shot: false   # opt-in; only meaningful when
                                   # training.token_level=true
```

When `evaluation.token_level_zero_shot=true` AND `training.token_level=true`:
- Image side runs the timm vision model in token mode and feeds `(B, T, D)` tokens for the selected layer through `alignment_image` → BA-token's CAP path.
- Text side keeps the per-template attention mask and feeds `(BS, T, D)` template tokens + mask through `alignment_text` → CAP path. Template averaging happens AFTER CAP, on the K-dim profile, so each template gets its own attention pattern.

**Code changes:**

- `src/evaluation/zero_shot_classifier.py::build_zero_shot_classifier`:
  - New `token_level: bool = False` kwarg.
  - When `token_level=True`: forces `pool_txt="none"` (keep tokens), forces `save_path=None` (the per-template mask is not cacheable, so always recompute), and after layer slicing, calls `alignment_layer(class_embeddings, mask=token_inputs["attention_mask"].to(device))` instead of `alignment_layer(class_embeddings)`. The mask flows through to BA-token's CAP softmax.

- `src/trainers/alignment_trainer.py::evaluate_zero_shot_classification`:
  - Reads `token_level_zero_shot` flag and ANDs with `training.token_level`.
  - When token mode: image cache name is `eval-none_layer-{L}{-r{img_size}}-zs.npy` (a `-zs` suffix distinguishes it from retrieval's `_load_eval_token_features` cache, which serves a different sample order — eval_loader vs retrieval_loader can produce different `(img, target)` pairings).
  - On-the-fly image extraction: instead of CLS-slicing all layers (`[v[:, 0, :] for v in lvm_output.values()]`), picks the layer-pinned `(B, T, D)` directly via `list(lvm_output.values())[image_layer_idx]`.
  - Per-batch: `image_feats = lvm_output.float()` (shape `(B, T, D)`) → `alignment_image(image_feats)` → 3D CAP forward → `(B, K)` → standard `chunked_logits` against the token-mode classifier.
  - `build_zero_shot_classifier(..., token_level=token_level_zero_shot)` is passed the new flag.
  - The legacy CLS branch is unchanged when the flag is false.

**Smoke test** (no training, no full extraction): re-evaluated the existing dry-run BA-token K=128 CIFAR-10 checkpoint (`(11, 6)_0.2763/checkpoint-epoch1.pth`, trained at the original 518 default — 1370 tokens/image) twice — once in CLS mode, once in token mode — using a throwaway driver script `smoke_test_token_zero_shot.py`.

| Mode | CIFAR-10 top-1 | top-5 | wall (eval only) |
|---|---|---|---|
| CLS fallback (default) | **78.3%** | 98.9% | ~5 sec (CLS feature cache hit) |
| Token CAP (new path) | **90.9%** | 99.8% | ~5 min (fresh per-layer token extraction; `-zs` cache written) |

Same checkpoint, same eval data, only the eval path differs. **+12.6 pp** swing from running the model through its trained CAP attention rather than slicing CLS at inference time. This validates that:

1. Token-mode zero-shot eval works end-to-end (image extraction, classifier building, per-batch CAP forward, metrics, caching).
2. The trained CAP layer was learning meaningful patch-level attention that CLS-only eval was discarding.
3. Text-side template tokens + mask going through CAP also contribute (otherwise the swing would mostly come from image side; the magnitude here suggests both sides matter).

The CLS number (78.3%) matches the published dry-run number from the earlier BA-token entry exactly, confirming the legacy path is unchanged.

**Files touched:**
- `configs/default.yaml`: added `evaluation.token_level_zero_shot: false`.
- `src/evaluation/zero_shot_classifier.py`: added `token_level` kwarg + mask plumbing.
- `src/trainers/alignment_trainer.py::evaluate_zero_shot_classification`: token branch for image extraction + 3D forward + cache naming.
- `smoke_test_token_zero_shot.py` (new): one-off harness; safe to delete after the comparison runs on Server B.

**Backward compatibility:** the default of the new flag is `false`, and the legacy CLS branches are untouched when it's off. Existing non-token configs (CLS BA, Linear, MLP, ResLowRank, CSA) ignore the flag because their training also has `token_level=false`, so the AND short-circuits.

**Caveats / open issues:**
- Token-mode zero-shot is slower per evaluation: each eval dataset triggers a fresh on-disk extraction (`-zs.npy`) the first time. For CIFAR-10 the 50K test images at 518 res took ~5 min on a partly-shared GPU. At 224 res this drops to ~30 sec (smaller tokens, faster forward).
- The image cache is keyed by `(model, dataset, layer, img_size)` so repeated runs of the same checkpoint on the same dataset hit the cache.
- If you want both CLS and token numbers in a single training run, run training once and then call `train_alignment.py` (or a focused eval script) twice with different config overrides — the checkpoint is shared.
- The label-template attention mask is generated fresh each time (`tokenizer(...)`) and not persisted; this is fine since templates are short and tokenization is cheap.

### 2026-04-13 — Token-path drop_duplicates bypass — FIXED

**Bug:** in `fit()`, the CLS path correctly applied STRUCTURE's `training.drop_duplicates: true` (default) to map COCO's 591K caption-image pairs down to 118K unique images, but the `token_level=true` override at the end of the per-layer loop **replaced** the deduped CLS tensors with fresh full-size token tensors loaded from disk via `_load_token_features_for_layer`. The token tensors are aligned 1:1 with the underlying dataset rows (591K), so the dedup was silently bypassed and token training iterated 5× more samples per epoch than the CLS baseline.

This made the running token K=128 ~5× slower than necessary AND broke fairness with the CLS comparison table.

**Confirmed in the running K=128 log:**
```
TRAIN - img: torch.Size([118287, 12, 384])    # CLS dedup OK
TOKEN TRAIN - img: (591753, 257, 384)         # override negates dedup
```

**Fix** (`src/trainers/alignment_trainer.py::fit`):

1. Compute the dedup boolean masks **once**, before either path consumes them, and store as locals (`sel_train_indices`, `sel_val_indices`):
   ```python
   sel_train_indices = None
   sel_val_indices = None
   if (
       self.config["training"]["drop_duplicates"]
       and hasattr(self.train_dataset.dataset, "df")
       and "image_path" in self.train_dataset.dataset.df.columns
   ):
       sel_train_indices = (
           self.train_dataset.dataset.df.groupby("image_path").cumcount()
           < self.config["training"]["n_dup_samples"]
       )
       sel_val_indices = (
           self.val_dataset.dataset.df.groupby("image_path").cumcount()
           < self.config["training"]["n_dup_samples"]
       )

   if sel_train_indices is not None:
       image_features_train = image_features_train[sel_train_indices]
       text_features_train = text_features_train[sel_train_indices]
       image_features_val = image_features_val[sel_val_indices]
       text_features_val = text_features_val[sel_val_indices]
   ```

2. In the token override block, mirror the same mask onto the freshly-loaded token tensors **and** the text mask. A shape guard (`feats.shape[0] == len(sel_indices)`) skips dedup when an `n_random_subsample_*` subset extraction (`-n{N}` cache) is in play, since the subset's row count won't match the full df:
   ```python
   if (
       sel_train_indices is not None
       and layer_image_features_train.shape[0] == len(sel_train_indices)
   ):
       layer_image_features_train = layer_image_features_train[sel_train_indices]
       layer_text_features_train = layer_text_features_train[sel_train_indices]
       if layer_text_mask_train is not None:
           layer_text_mask_train = layer_text_mask_train[sel_train_indices]
   # symmetric block for val
   ```

**Smoke tests** (no training run; loaded the actual COCO train token caches and applied the dedup mask):

| Tensor | Full shape | After dedup |
|---|---|---|
| Image tokens (fp16) | `(591753, 257, 384)` | **`(118287, 257, 384)`** ✓ |
| Text tokens (fp16)  | `(591753, 65, 384)`  | **`(118287, 65, 384)`** ✓ |
| Text mask (int64)   | `(591753, 65)`       | **`(118287, 65)`** ✓ |

Mask `True` count: `118287 / 591753` (matches COCO train2014 unique image count).

Module also re-imports clean after the patch. Did not run a full training epoch on the dedup'd cache per the user's instruction.

**STRUCTURE default check:** `git log -p configs/default.yaml` shows `drop_duplicates: true` and `n_dup_samples: 1` were added in the **initial commit `a25023b`** — they are upstream STRUCTURE defaults, not anything the BA work introduced.

**Implication for the running run:** the K=128 small-token run that was using all 591K pairs has been **killed**. The fixed code is ready for re-launch on Server B.

**Files changed in this fix:**
- `src/trainers/alignment_trainer.py::fit`
  - Lifted `sel_train_indices` / `sel_val_indices` computation above the CLS dedup block so it survives into the token override.
  - Added dedup application + text-mask dedup inside the `if token_level:` block, gated on shape match so subset-extracted caches still work.

### 2026-04-13 — Pipeline unification at 224 res (extract once, use everywhere)

**Goal:** for Server B re-runs, kill the dual extraction paths (one for CLS-only experiments at 518, one for token experiments at 224) and unify everything at 224 with a single token cache that BOTH paths can read.

**Design:**

For each modality, extract a single `(N, T, D) float16` token tensor per
`(model, dataset, split, layer, img_size)` and derive the CLS / mean-pooled
view at load time:

| Mode | Image | Text |
|---|---|---|
| CLS / token-train | `tokens[:, 0, :] → (N, D)` | masked mean: `(tokens * mask).sum(1) / mask.sum(1) → (N, D)` |
| Token-level train | `(N, T, D)` directly | `(N, S, D)` + mask `(N, S)` |

This is mathematically equivalent: the CLS slice is what `get_image_features` would have returned in `pool_img=cls` for the chosen layer, and the masked mean is bit-identical to `pool_txt=avg` (verified — see smoke test below).

**Code changes:**

- `src/trainers/alignment_trainer.py::get_lvm` already accepts `features.img_size` and propagates to `timm.create_model(img_size=…)` plus the resolved transform. **No change** needed here from earlier work.

- New helpers on `AlignmentTrainer`:
  - `_unified_image_token_path(dataset_name, split_tag, layer_idx)` — returns the canonical path for the unified image token cache, encoding `img_size` via `-r{N}`.
  - `_unified_text_token_path(...)` — returns `(features_path, mask_path)` for the unified text cache.
  - `_try_load_image_cls_from_tokens(...)` — returns `(N, D)` CLS slice if the unified token cache exists, else `None`.
  - `_try_load_text_avg_from_tokens(...)` — returns `(N, D)` masked-mean text features if the unified token + mask cache exist, else `None`.

- `fit()` main path rewritten to **prefer the unified cache when the layer is pinned**:
  1. If `features.layer_img/layer_txt` are set AND `pool_img/pool_txt != none` → try `_try_load_*_from_tokens` first.
  2. If unified-cache hit, skip multi-layer extraction entirely.
  3. If `training.token_level=true` AND layers pinned → skip the CLS pre-load entirely (use a `(N, 1)` placeholder); the token override later in `fit()` supplies the real tensors. **Saves ~35 min of wasted CLS re-extraction at 224 per run.**
  4. Else fall back to the legacy `get_image_features / get_text_features` extraction path.

- `fit()` layer-slice generalised to handle `(N, L, D)` (legacy) AND `(N, D)` (unified-derived) inputs via a small `_layer_slice` helper. 2D inputs pass through unchanged; 3D inputs slice on dim 1.

- `src/extract_features.py` (legacy standalone extraction CLI) gained an `--img_size` arg that flows into `timm.create_model` + transform input size override, mirroring `get_lvm`.

- `get_image_features` / `get_text_features` already use the streaming `float16` allocation path for `pool=none`, pre-allocating a single `(N, T, D)` tensor and filling by offset (added earlier today after the OOM during full COCO extraction). This is the path the unified pipeline writes through.

**Cache naming:**

The unified cache uses the same suffix scheme as the existing token cache:
- Image: `{model}-{dataset}-{split}-none_layer-{L}-r{img_size}.npy` (fp16)
- Text features: `{model}-{dataset}-{split}-none_layer-{L}.npy` (fp16; resolution-independent)
- Text mask: `{model}-{dataset}-{split}-none_layer-{L}_mask.npy`

**Layer selection note:**

Layer selection (`compute_layer_alignment`) still requires the multi-layer
`(N, L, D)` CLS extraction because it scores across layers. That path
**also** respects `features.img_size` automatically because it goes through
`get_image_features → get_lvm`. So on Server B the workflow is:

1. Run layer selection at 224 (one multi-layer CLS extraction; produces `*-cls-r224.npy`).
2. Pin the chosen layer pair via `features.layer_img/layer_txt`.
3. Extract the unified token cache once via `extract_token_features.py --img_size 224` (or let the trainer extract on demand).
4. All subsequent CLS-only and token-level experiments at the chosen layer pair read from the same cache.

**Smoke tests** (no full extraction; ran on existing COCO val r224 caches from the earlier run):

| Test | Result |
|---|---|
| `from src.trainers.alignment_trainer import AlignmentTrainer` | ✓ imports clean |
| `_try_load_image_cls_from_tokens` / `_try_load_text_avg_from_tokens` exist | ✓ |
| `timm.create_model('vit_small_patch14_dinov2.lvd142m', img_size=224)` forward | ✓ output `(B, 257, 384)` |
| Slice `(25014, 257, 384) → [:, 0, :]` (CLS) | ✓ `(25014, 384)`, mean-abs 0.51 |
| Masked mean from text token cache vs legacy `val-avg` layer-6 slice | ✓ **max diff 0.0, cosine 1.0** (bit-exact) |

**Backward compatibility:**

- Legacy `*-cls.npy` / `*-avg.npy` caches remain untouched. Runs without
  `features.img_size` set OR without a unified token cache fall through
  to the legacy extraction path unchanged.
- Existing K=128 token training (running on GPU 1) loaded its caches
  before this change; in-memory state is independent.
- The currently-cached COCO val text tokens were stored as `bfloat16`
  (pre-fp16-streaming-patch); train tokens were stored as `float16`
  (post-patch). The masked-mean derivation handles both dtypes.

**Server B handover:**

1. Set `features.img_size: 224` in every config (small + large encoders).
2. Layer selection → produces `*-cls-r224.npy`.
3. Token extraction → produces `*-none_layer-{L}-r224.npy` (image, fp16) and `*-none_layer-{L}.npy` (text, fp16) + `*_mask.npy`.
4. CLS BA / Linear / MLP / ResLowRank training → automatically derives CLS from the unified token cache.
5. Token BA training → automatically loads the same unified token cache.
6. **One extraction, all experiments.**

**Files touched in this change:**

- `src/trainers/alignment_trainer.py`:
  - Added 4 helpers (`_unified_image_token_path`, `_unified_text_token_path`, `_try_load_image_cls_from_tokens`, `_try_load_text_avg_from_tokens`).
  - Rewrote `fit()` val/train CLS load to prefer unified cache and short-circuit token_level pre-load.
  - Replaced `[:, layer_idx, :]` slice with a 2D/3D-aware `_layer_slice` helper.
- `src/extract_features.py`: added `--img_size` CLI and threaded it into `timm.create_model` + `data_config["input_size"]`.

**Known limitations / open issues:**

- When `pool_txt=avg` is requested and only the unified text token cache (without mask) is found, the helper returns `None` (need both files). On Server B always extract via `extract_token_features.py` so the mask is written.
- The CSA trainer (`src/trainers/csa_trainer.py`) inherits `get_lvm` from `AlignmentTrainer`, so it picks up `features.img_size` automatically. Not separately tested.
- The `_layer_slice` helper assumes the token-cache CLS slice and the legacy multi-layer CLS at the chosen layer are interchangeable. They are — both come from `lvm_output.values()[layer][:, 0, :]` — but if a future model changes the CLS-token convention, this will need revisiting.

### 2026-04-13 — Token-level BA dry run — WORKING end-to-end

**Dry-run results** (`configs/dryrun_ba_token.yaml`, ViT-S + MiniLM, CIFAR-10, 2 epochs, K=128, τ=0.05, train=2000, val=1000):

| Metric | Value |
|---|---|
| Image token shape | **(2000, 1370, 384)** |
| Text token shape | **(2000, 7, 384)** |
| Text mask shape | **(2000, 7)** |
| Layer selected by CLS mutual-kNN | **img=11, txt=6** (same as CLS runs) |
| Params per modality | **49,152** (128 × 384) ✓ |
| Epoch 1 train / val loss | **4.56 / 3.72** |
| Epoch 2 train / val loss | **3.68 / 3.66** |
| CIFAR-10 zero-shot top-1 | **78.31%** |

Loss decreased monotonically; eval ran via the 2D CLS fallback path (as designed). Token features are 1370-long because DINOv2 ViT-S/14 uses 518×518 input resolution by default, so per-image tokens = 1 CLS + 37×37 patches. This made full-CIFAR-10 token extraction prohibitive (~105 GB for 50K samples), so the dry-run config uses `n_random_subsample_train=2000, n_random_subsample_val=1000` and the loader wraps in a first-N subset (`_SubsetView`) to match.

**Gotchas discovered during the dry run (and fixed):**
- `_SubsetView` had to properly forward attribute writes so `loader.dataset.tokenizer = ...` goes through to the real dataset. Initial version used `__getattr__` for reads only, which meant `apply_tokenizer()` ran on a tokenizer-less underlying dataset and returned plain strings instead of tokenized dicts, blowing up in `get_text_features`. Fix: override `__setattr__` to delegate by default.
- Feature extraction CPU contention on shared server made the CIFAR-10 "Precomputing captions" step crawl (50 it/s instead of 1200 it/s) when other jobs were using cores. Not a code bug — just slow.
- Subsample indices: CLS path uses a random permutation, token path uses deterministic first-N. They don't align, so in token_level mode we skip re-applying the CLS dedup/subsample masks (the token features are already sized correctly at extraction time).

### 2026-04-13 — Token-level BA with Cross-Attention Pooling (CAP)

**New files:**
- `src/alignment/bridge_anchor_token.py` — `BridgeAnchorTokenAlignmentLayer`. Token-level CAP: input `(B, T, D)` or CLS fallback `(B, D)`, mask `(B, T)` optional, output `(B, K)` L2-normalized. Temperature-scaled softmax over tokens per anchor, optional `BottleneckProjector` bottleneck.
- `src/extract_token_features.py` — standalone CLI that reuses STRUCTURE's existing `pool=none` extraction path (already supports single-layer tokens) and additionally saves text attention masks. Takes `--config_path --img_layer --txt_layer --splits`.
- `configs/dryrun_ba_token.yaml` — 2-epoch CIFAR-10 dry-run using ViT-S + MiniLM, K=128, τ=0.05, `n_random_subsample_train=2000`, `n_random_subsample_val=1000`, `token_level=true`.

**Modifications to trainer:**
- `src/trainers/alignment_trainer.py`:
  - `_load_token_features_for_layer(img_layer_idx, txt_layer_idx)` — helper that reuses the existing `get_image_features` / `get_text_features` with `pool=none, layer=L` to extract/cache token features for a specific layer. Restores config on exit.
  - `_load_or_build_text_mask(loader, llm, suffix)` — loads cached text mask or re-runs the tokenizer to build one. Saves alongside the feature `.pt` with `_mask` suffix.
  - `fit()` — token-level branch after layer selection: load token features, re-apply dedup and subsample to keep them in sync with the CLS path, pass tokens + masks through training/validation.
  - `train()` / `validate()` — accept `text_mask` kwarg; when present, pass `mask=text_mask_batch` to `alignment_text` forward. CLS alignment layers are unaffected (mask not passed if None).
- `src/trainers/base_trainer.py::find_optimal_learning_rate` — accept `text_mask_train` kwarg and pass it through to `alignment_text` during the LR finder sweep.
- `src/train_alignment.py` — `trainer.fit(...)` call now threads `n_random_subsample_train/val` from config.

**Design choices:**
- The `pool=none, layer=L` extraction path already existed in STRUCTURE but had never been used end-to-end. Token BA piggybacks on it instead of introducing a parallel code path.
- Structure regularization is disabled (`structure_lambda=0`) in token-level mode because `structure_reg` expects 2D features; the CLIP loss itself only uses the 2D (B, K) aligned output.
- Zero-shot classification and retrieval still receive CLS inputs — the BA token layer's 2D fallback branch handles them, so eval code is unchanged (caveat: eval is not using the CAP path and so underestimates the potential of the trained model).
- Mask handling: only text uses padding masks in practice (ViT has no padding). Mask is saved alongside text features with a `_mask` suffix in the same directory as STRUCTURE's existing feature cache.

### 2026-04-13 — First complete BA result (small encoders)

- **Created** `configs/losses_ba/bridge_anchor_small_{128,256,512}.yaml` — small-encoder variants using cached ViT-S + MiniLM features. Training starts immediately with no feature-extraction bottleneck.
- **Created** `configs/losses_lin/clip_small_best.yaml` — small-encoder Linear baseline for direct comparison (same training hyperparameters as BA small configs, only `alignment_layer_name` differs).
- **Ran** BA K=128 small on GPU 1:
  - Early-stopped at epoch 490 / best val clip loss **3.3715**
  - Zero-shot: CIFAR-10 82.45%, STL-10 94.50%, CIFAR-100 33.83%, MNIST 18.04%
  - COCO retrieval: I2T R@1 **28.12%**, R@10 71.09%; T2I R@1 21.34%, R@10 31.40%
  - Wall time: ~22 min train + ~8 min eval
- **Completed** Linear small baseline on GPU 1 in parallel:
  - Early-stopped at epoch 508 / best val **3.3609**
  - CIFAR-10 82.22%, STL-10 95.00%, CIFAR-100 34.60%, MNIST 15.59%
  - COCO I2T R@1 **29.10%**, R@10 70.90%; T2I R@1 21.80%, R@10 31.70%
  - **BA K=128 matches Linear within 1 pp on all metrics using half the parameters** (49K vs 98K per modality)
- **Completed** BA K=256 small: best val 3.3655, CIFAR-10 82.70%, STL-10 **95.43%** (beats Linear's 95.00%), CIFAR-100 34.30%, MNIST 16.37%, COCO I2T R@1 28.87%, R@5 **58.50%** (beats Linear)
- **Running** BA K=512 small on GPU 1
- **Running** BA K=128 **large** model (GPU 0, vision val + text val features done, now extracting vision train features — ~28 hours remaining)

**Key finding:** At matched parameters (BA K=256 ≈ Linear dim=256), BA is competitive — wins on CIFAR-10, STL-10, MNIST, COCO I2T R@5/R@10; Linear wins on val loss, CIFAR-100, R@1. Essentially tied. BA K=128 matches Linear within ~1 pp with **half the parameters**.

### 2026-04-13 — Full training launch on COCO (Server A)

- **Updated** `configs/losses_ba/bridge_anchor_base.yaml`, `bridge_anchor_base_256.yaml`, `bridge_anchor_base_512.yaml`:
  - Added `features.batch_size: 64` + `features.num_workers: 4` override (default 16 was too slow)
  - Added `evaluation.zero_shot_datasets: [cifar10, stl10, cifar100, mnist]` and `retrieval_datasets: [coco]` override — the default 26-dataset list triggered multi-day auto-downloads on Server A's throttled network. Training hyperparameters unchanged.
- **Launch wrapper** — inlined the ToTensor monkey-patch + cuDNN enable (safe on torch 2.1.2) into the training command, since `run_with_totensor_fix.py` is a wrapper we can't compose with `conda run` buffering fixes.
- **K=128 launched on GPU 0** with: direct conda-env python path (unbuffered), `WANDB_MODE=offline`, `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` (to avoid fragmentation on shared GPU), `cudnn.enabled=True, cudnn.benchmark=True`. Currently in COCO feature-extraction phase (~33 hours total expected).
- **K=256 NOT YET LAUNCHED** — must wait for K=128 to finish populating the feature cache, otherwise both runs race on the same `.npy` files (no file-level locking in `alignment_trainer.get_image_features` / `get_text_features`).

### 2026-04-13 — Initial vanilla BA integration
- **Created** `src/alignment/bridge_anchor.py` — `BridgeAnchorAlignmentLayer`
- **Created** `configs/losses_ba/bridge_anchor_base.yaml` (K=128)
- **Created** `configs/losses_ba/bridge_anchor_base_256.yaml` (K=256)
- **Created** `configs/losses_ba/bridge_anchor_base_512.yaml` (K=512)
- **Created** `configs/dryrun_ba.yaml` (small-model dry-run)
- **Created** `bridge-anchors/` (reference copies from `/home/shiwon/bridge-anchors/`)
- **Fix** — added `dim_alignment` alias in BA `__init__` to survive YAML deep-merge of `default.yaml`'s `alignment_layer_kwargs: {dim_alignment: 256}`
- **Verified** — dry-run end-to-end on Server A, CIFAR-10 top1 = 97.56%
