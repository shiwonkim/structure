# DINOv2 ↔ CLIP Vision-Vision Alignment via BridgeAnchors

## Concept

BridgeAnchors (BA) is a general representation bridging framework. Beyond cross-modal (vision-language) alignment, BA can align two vision encoders with complementary strengths:

- **DINOv2**: excellent spatial/structural features, no language understanding
- **CLIP**: language-aligned features, weaker spatial structure

After BA alignment, DINOv2 features can be projected into CLIP's text-aligned space, gaining zero-shot language capabilities without any text training.

## Architecture

```
Training (paired images, no text):
  DINOv2 tokens (257×D_dino) → BridgeAnchorToken(D_dino, K) → K-dim profile
  ↔ contrastive loss ↔
  CLIP vision tokens (257×D_clip) → BridgeAnchorToken(D_clip, K) ��� K-dim profile

Text bridge (after vision-vision training):
  CLIP vision projected (D_proj) → BridgeAnchor(D_proj, K) → K-dim profile
  Trained to match: align_clip(CLIP_vision_tokens) → K-dim target

Inference (zero-shot classification):
  DINOv2 image tokens → align_dino → K-dim profile
  CLIP text projected  → align_text → K-dim profile
  Compare in K-dim space → class prediction
```

## Experiment Configurations

### Server A: DINOv2 ViT-B/14 ↔ CLIP ViT-L/14

| Component | Model | Hidden dim | Tokens | Projection |
|---|---|---|---|---|
| DINOv2 | `vit_base_patch14_dinov2.lvd142m` | 768 | 257 (16×16+CLS) | — |
| CLIP vision | `openai/clip-vit-large-patch14` | 1024 | 257 (16×16+CLS) | 768 |
| CLIP text | (same model) | 768 | varies | 768 |

- `align_dino`: `BridgeAnchorTokenAlignmentLayer(input_dim=768, K=512)`
- `align_clip`: `BridgeAnchorTokenAlignmentLayer(input_dim=1024, K=512)`
- `align_text`: `BridgeAnchorAlignmentLayer(input_dim=768, K=512)` (text bridge)

**Script**: `scripts/dino_clip_token_alignment.py`
**Status**: Running on Server A GPU 1

### Server B: DINOv2 ViT-L/14 ↔ CLIP ViT-L/14

| Component | Model | Hidden dim | Tokens | Projection |
|---|---|---|---|---|
| DINOv2 | `vit_large_patch14_dinov2.lvd142m` | 1024 | 257 (16×16+CLS) | — |
| CLIP vision | `openai/clip-vit-large-patch14` | 1024 | 257 (16×16+CLS) | 768 |
| CLIP text | (same model) | 768 | varies | 768 |

- `align_dino`: `BridgeAnchorTokenAlignmentLayer(input_dim=1024, K=512)`
- `align_clip`: `BridgeAnchorTokenAlignmentLayer(input_dim=1024, K=512)`
- `align_text`: `BridgeAnchorAlignmentLayer(input_dim=768, K=512)` (text bridge)

## Code Changes for Server B

Copy `scripts/dino_clip_token_alignment.py` and modify:

### 1. DINOv2 model name
```python
# Server A (ViT-B):
"vit_base_patch14_dinov2.lvd142m"
# Server B (ViT-L):
"vit_large_patch14_dinov2.lvd142m"
```

### 2. DINOv2 feature cache paths
```python
# Server A:
feat_dir / "vit_base_patch14_dinov2.lvd142m-CocoCaptionDataset-train-none_layer-11-r224.npy"
# Server B:
feat_dir / "vit_large_patch14_dinov2.lvd142m-CocoCaptionDataset-train-none_layer-23-r224.npy"
```
Note: ViT-L has 24 layers, so last layer is 23 (vs 11 for ViT-B with 12 layers).
If features aren't cached yet, run the standard STRUCTURE feature extraction first.

### 3. DINOv2 input dimension
```python
# Server A:
align_dino = BridgeAnchorTokenAlignmentLayer(input_dim=768, ...)
# Server B (both DINOv2 and CLIP are 1024):
align_dino = BridgeAnchorTokenAlignmentLayer(input_dim=1024, ...)
```

### 4. DINOv2 layer index for eval
```python
# Server A:
feats = list(dino_vision(images).values())[11]   # layer 11
# Server B:
feats = list(dino_vision(images).values())[23]   # layer 23
```

### 5. CLIP features (no change needed)
CLIP ViT-L/14 is the same model on both servers. Feature extraction code is identical.

## Pipeline Summary

1. **Extract CLIP ViT-L/14 features** on COCO train+val (~3 hours for ViT-L)
   - Hidden tokens: (N, 257, 1024) — for vision-vision alignment
   - Projected CLS: (N, 768) — for text bridge training
   - Deduplicate: COCO has 5 captions/image; DINOv2 features are already deduped

2. **Phase 1: Vision-Vision alignment** (~3-5 hours)
   - Contrastive loss on paired DINOv2 and CLIP token features
   - Early stopping with patience 200

3. **Phase 2: Text bridge** (~30 min)
   - Train CLS-level BA to map CLIP projected (768) → K-dim
   - MSE loss to match token-level CLIP profiles
   - Early stopping with patience 100

4. **Phase 3: Zero-shot eval** (~10 min)
   - DINOv2 → align_dino → K-dim
   - CLIP text → align_text → K-dim
   - Cosine similarity → classification

## Expected Results

DINOv2 (no language pretraining) should achieve meaningful zero-shot classification
accuracy through the BA anchor bridge to CLIP's text space. The ViT-L↔ViT-L pairing
(Server B) should outperform ViT-B↔ViT-L (Server A) since both sides are 1024-dim
with no dimension mismatch.

## Training Data

Uses COCO 2014 train (82,783 images) for training and COCO val (40,504) for validation.
No text is used during BA training — only paired image features from both encoders.

## Evaluation

- CIFAR-10, CIFAR-100, STL-10 zero-shot classification
- Compare with: CLIP direct ZS (upper bound), DINOv2 linear probe (reference)
