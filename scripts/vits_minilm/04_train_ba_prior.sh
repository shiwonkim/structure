#!/bin/bash
# NOTE: First run extracts CLS-attention from DINOv2's last block
# (~4 min for the dedup'd 82K COCO train split + 40K val split on ViT-S)
# and caches to results/features/*_cls_attn_layer-<L>-r224.npy. Subsequent
# runs reuse the cache and skip straight to training.
# Sequential Token BA + CLS attention prior runs: K=128/256/512.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../common.env"
source "$SCRIPT_DIR/config.env"

GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU

cd "$PROJECT_ROOT"
mkdir -p logs
LOG="logs/${ENCODER_TAG}_ba_prior_$(date +%Y%m%d_%H%M).log"
START=$(date +%s)

echo "[$(date)] Token BA + CLS attention prior: ${ENCODER_TAG} on GPU ${GPU}" | tee "$LOG"

CONFIGS=(
    "01_token_ba_k128_prior  configs/ba/vits_minilm/token_k128_prior.yaml"
    "02_token_ba_k256_prior  configs/ba/vits_minilm/token_k256_prior.yaml"
    "03_token_ba_k512_prior  configs/ba/vits_minilm/token_k512_prior.yaml"
)

COMPLETED=()
for entry in "${CONFIGS[@]}"; do
    read -r NAME CFG <<< "$entry"
    if [[ ! -f "$CFG" ]]; then
        echo "[SKIP] $NAME — missing config $CFG" | tee -a "$LOG"
        continue
    fi
    echo "[$(date)] >>> $NAME ($CFG)" | tee -a "$LOG"
    if ! PYTHONPATH=. python src/train_alignment.py --config_path "$CFG" 2>&1 | tee -a "$LOG"; then
        echo "[WARN] $NAME failed. Continuing to next run." | tee -a "$LOG"
    fi
    COMPLETED+=("$NAME")
done

echo "[$(date)] Done. Completed: ${COMPLETED[*]:-none}. Wall: $(( $(date +%s) - START ))s" | tee -a "$LOG"
