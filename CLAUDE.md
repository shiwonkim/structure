# Bridge Anchors x STRUCTURE Integration

## Project Overview
Integrating BridgeAnchors (BA) — a novel cross-modal alignment method using learnable anchor points with cross-attention pooling — into the STRUCTURE codebase for fair comparison with existing alignment methods (Linear, MLP/ResLowRankHead, CSA).

## Repository Structure
- This repo is a clone of https://github.com/mlbio-epfl/STRUCTURE
- Our BA method will be added as a new alignment method alongside existing ones
- Original BA codebase is on Server B at [path to original BA code]

## Environment
- Python 3.10, PyTorch 2.1.2+cu118, CUDA 11.8
- Conda env: `structure`
- Key packages: timm 0.9.16, transformers 4.45.2, deepspeed 0.14.4
- Docker image will be built from Server A, deployed on Server B (CUDA driver 550, CUDA 12.4)

## Data
- COCO: symlinked at ./data/COCO/ (train2014->train2017, val2014->val2017 from /home/data/2026_COCO)
- Additional datasets needed: Flickr30k (retrieval), ImageNet (classification)
- Most zero-shot datasets auto-download via torchvision

## STRUCTURE Pipeline
1. Feature extraction: frozen encoders -> per-layer features cached to disk
2. Layer selection: mutual kNN finds best (layer_img, layer_txt) pair
3. Alignment training: lightweight alignment layer trained on cached features
4. Evaluation: zero-shot classification (22 datasets) + retrieval (Flickr30k, COCO)

## Default Encoders
- Vision: vit_large_patch14_dinov2.lvd142m (DINOv2 ViT-L, 24 blocks, 1024-dim)
- Language: sentence-transformers/all-roberta-large-v1 (RoBERTa-Large, 25 layers, 1024-dim)

## Adding a New Alignment Method
1. Create file in src/alignment/
2. Subclass BaseAlignmentLayer (requires __init__(input_dim) + forward(z) -> z)
3. Decorate with @AlignmentFactory.register()
4. Import in src/alignment/__init__.py
5. Reference by class name in config: training.alignment_layer_name

## Key Differences from Our BA Codebase
- STRUCTURE caches all features to disk first, then trains alignment only (no encoder forward during training)
- 1000 epochs is standard (each epoch is very fast with cached features)
- Uses AdamW with betas=[0.9, 0.95], weight_decay=1e-4
- Cosine schedule with T_max = 50 epoch cycles (scheduler_epoch_cycles: 50)
- Early stopping patience=200 epochs
- BS=4096, uses LR finder
- Alignment layer interface: input (B, D) -> output (B, D_align), single modality at a time

## Known Issues & Workarounds
- CocoCaptionDataset.load_image() returns tensors, but timm transforms include ToTensor() which rejects tensors. Use run_with_totensor_fix.py wrapper for fresh COCO feature extraction.
- cca_zoo 2.5.0 PCA does not filter zero-variance components, causing NaN in CSA with low-rank features. Use run_csa_fix.py wrapper for CSA. Default large models (1024-dim) are unaffected.
- Server A CUDA driver 470 requires torch 2.1.2 (not 2.2.0) due to cuDNN 8.7 incompatibility. Server B driver 550 should work with either.

## Development Rules
- Always update PROJECT_LOG.md after completing any task
- Don't modify existing STRUCTURE source code unless necessary — add new files
- Test changes with dry run before full experiments
- Use wandb for experiment tracking
