#!/usr/bin/env bash
# Batch 4: Token-level Linear and MLP baselines (ViT-S + MiniLM).
# Reuses the existing token cache from Batch 2 (T=257, CLS included).
# Usage: bash scripts/vits_minilm/05_train_token_baselines.sh <gpu_id>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

source scripts/common.env
source scripts/vits_minilm/config.env

GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="${PYTHONPATH:-.}"

PY=/home/shiwon/miniconda3/envs/structure/bin/python
LOGFILE="logs/vits_minilm_token_baselines_$(date '+%Y%m%d_%H%M').log"
mkdir -p logs

CONFIGS=(
  "01_linear_token         configs/linear/vits_minilm/linear_d512_token.yaml"
  "02_linear_token_struct  configs/linear/vits_minilm/linear_d512_token_struct.yaml"
  "03_mlp_token            configs/mlp/vits_minilm/mlp_d512_token.yaml"
  "04_mlp_token_struct     configs/mlp/vits_minilm/mlp_d512_token_struct.yaml"
)

COMPLETED=()
exec > >(tee -a "$LOGFILE") 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Batch 4 token baselines — GPU $GPU"
echo "Log: $LOGFILE"

for entry in "${CONFIGS[@]}"; do
  read -r label cfg <<< "$entry"
  echo ""
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $label ($cfg)"
  if $PY -u src/train_alignment.py --config_path "$cfg"; then
    COMPLETED+=("$label")
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  $label (success)"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAIL  $label (exit=$?)"
  fi
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done. Completed: ${COMPLETED[*]}. Wall: ${SECONDS}s"
