#!/bin/bash
# NOTE: First run per encoder combo will:
#   1. Extract all-layer CLS features for layer selection (~10 min)
#   2. Run mutual kNN layer selection to find best (img_layer, txt_layer) pair
#   3. Extract token features for the selected layer (~30 min for ViT-S)
#   4. Begin training
# Subsequent runs reuse cached features and skip extraction.
# Sequential FreezeAlign runs for ViT-L + RoBERTa-Large.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../common.env"
source "$SCRIPT_DIR/config.env"

GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU

cd "$PROJECT_ROOT"
mkdir -p logs
LOG="logs/${ENCODER_TAG}_freezealign_$(date +%Y%m%d_%H%M).log"
START=$(date +%s)

echo "[$(date)] FreezeAlign: ${ENCODER_TAG} on GPU ${GPU}" | tee "$LOG"

CONFIGS=(
    "01_freeze_align         configs/freezealign/vitl_roberta/fa_d512.yaml"
    "02_freeze_align_struct  configs/freezealign/vitl_roberta/fa_d512_struct.yaml"
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
        echo "[WARN] $NAME failed (possibly OOM). Retry with BS=2048 in the config." | tee -a "$LOG"
    fi
    COMPLETED+=("$NAME")
done

echo "[$(date)] Done. Completed: ${COMPLETED[*]:-none}. Wall: $(( $(date +%s) - START ))s" | tee -a "$LOG"
