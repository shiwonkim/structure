# STRUCTURE-BA — Docker Environment

Environment-only image for the STRUCTURE + BridgeAnchors integration project. Contains Python 3.10, PyTorch 2.1.2+cu118, and all dependencies. Code and data are mounted at runtime via volumes.

## Quick Setup (New Server)

```bash
bash docker/setup_server.sh /path/to/STRUCTURE /path/to/data
```

## Build (from source)

```bash
cd docker && docker build -f Dockerfile.structure -t structure-ba:v1 .
```

## Run (manual)

```bash
docker run --gpus all -it --shm-size=16g \
  -v /path/to/STRUCTURE:/workspace/STRUCTURE \
  -v /path/to/data:/workspace/data \
  -e PYTHONPATH=/workspace/STRUCTURE \
  -w /workspace/STRUCTURE \
  structure-ba:v1 bash
```

## Notes
- Base image is `pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel` (includes nvcc, required by DeepSpeed import).
- Model weights (DINOv2, RoBERTa, etc.) are NOT pre-cached in the image. They will be downloaded on first feature extraction run. After extraction, cached features are saved to disk and reused.
- CUDA 11.8 is forward-compatible with CUDA 12.x host drivers (Server B driver 550).
- Claude Code is pre-installed for interactive development.
- Server A (NVIDIA driver 470): disable cuDNN with `torch.backends.cudnn.enabled = False` or use `run_dryrun.sh`.
