#!/usr/bin/env bash
# STRUCTURE-BA — New server setup script
# Usage: bash docker/setup_server.sh [code_dir] [data_dir]

set -euo pipefail

IMAGE="shiwonkim/structure-ba:v1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${1:-$(dirname "$SCRIPT_DIR")}"
DATA_DIR="${2:-${CODE_DIR}/data}"

echo "=== STRUCTURE-BA Server Setup ==="
echo "Image:    ${IMAGE}"
echo "Code dir: ${CODE_DIR}"
echo "Data dir: ${DATA_DIR}"
echo ""

if [ ! -d "$CODE_DIR" ]; then
    echo "ERROR: Code directory not found: ${CODE_DIR}"
    exit 1
fi

if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found."
    exit 1
fi

echo "Pulling ${IMAGE}..."
docker pull "${IMAGE}"
echo ""

echo "GPU sanity check..."
docker run --gpus all --rm "${IMAGE}" python -c \
    "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"
echo ""

echo "Launching container..."
docker run --rm --gpus all -it --shm-size=16g \
    --name structure-ba \
    -v "${CODE_DIR}":/workspace/STRUCTURE \
    -v "${DATA_DIR}":/workspace/data \
    -e PYTHONPATH=/workspace/STRUCTURE \
    -w /workspace/STRUCTURE \
    "${IMAGE}" bash
