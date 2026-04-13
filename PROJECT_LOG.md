# BridgeAnchors x STRUCTURE Integration — Project Log

## Current Status & Next Steps

### Infrastructure
- Server A: STRUCTURE environment set up (conda env: structure), dry run complete
- Server B: Original BA codebase, experiments running
- Docker: image built and verified on Server A (shiwonkim/structure-ba:v1, 19.5GB)

### Immediate TODO
1. ~~Build Docker image on Server A (cu118, all dependencies verified)~~ DONE
2. ~~Push to Docker Hub: `docker push shiwonkim/structure-ba:v1`~~ DONE (2026-04-12)
3. Deploy to Server B: `bash docker/setup_server.sh /path/to/STRUCTURE /path/to/data`
4. Implement BA as alignment method in STRUCTURE framework
5. Run fair comparison experiments

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
- CocoCaptionDataset has a ToTensor double-application bug; use run_with_totensor_fix.py wrapper
- cca_zoo 2.5.0 has PCA numerical instability; use run_csa_fix.py wrapper for CSA

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
