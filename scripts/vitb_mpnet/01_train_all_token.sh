#!/usr/bin/env bash
# ViT-B + mpnet-base: token-level BA (K=512,256,128) then baselines.
# First run will extract features + do layer selection (~15-20 min).
# Subsequent runs reuse the cache.
# Usage: bash scripts/vitb_mpnet/01_train_all_token.sh <gpu_id>

set -euo pipefail

cd /home/shiwon/STRUCTURE

source scripts/common.env

GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="${PYTHONPATH:-.}"

PY=/home/shiwon/miniconda3/envs/structure/bin/python
LOGFILE="logs/vitb_mpnet_all_token_$(date '+%Y%m%d_%H%M').log"
mkdir -p logs

CONFIGS=(
  "01_token_ba_k512      configs/ba/vitb_mpnet/token_k512.yaml"
  "02_token_ba_k256      configs/ba/vitb_mpnet/token_k256.yaml"
  "03_token_ba_k128      configs/ba/vitb_mpnet/token_k128.yaml"
  "04_linear_token       configs/linear/vitb_mpnet/linear_d512_token.yaml"
  "05_mlp_token          configs/mlp/vitb_mpnet/mlp_d512_token.yaml"
  "06_freezealign        configs/freezealign/vitb_mpnet/fa_d512.yaml"
)

COMPLETED=()
exec > >(tee -a "$LOGFILE") 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ViT-B + mpnet-base token-level suite — GPU $GPU"
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
