#!/bin/bash
# NOTE: First run per encoder combo will:
#   1. Extract all-layer CLS features for layer selection (~10 min)
#   2. Run mutual kNN layer selection to find best (img_layer, txt_layer) pair
#   3. Extract token features for the selected layer (~30 min for ViT-S)
#   4. Begin training
# Subsequent runs reuse cached features and skip extraction.
# Sequential CLS baseline runs: Linear, Linear+STR, MLP, MLP+STR, CSA, CSA+STR.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../common.env"
source "$SCRIPT_DIR/config.env"

GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU

cd "$PROJECT_ROOT"
mkdir -p logs
LOG="logs/${ENCODER_TAG}_baselines_$(date +%Y%m%d_%H%M).log"
START=$(date +%s)

echo "[$(date)] Baselines: ${ENCODER_TAG} on GPU ${GPU}" | tee "$LOG"

# Config paths (these must exist — either committed to configs/ or created
# from templates). For vits_minilm they're already in the repo.
CONFIGS=(
    "01_linear_d${DIM_ALIGNMENT}          configs/linear/vits_minilm/linear_d512.yaml"
    "02_linear_d${DIM_ALIGNMENT}_struct   configs/linear/vits_minilm/linear_d512_struct.yaml"
    "03_mlp_d${DIM_ALIGNMENT}             configs/mlp/vits_minilm/mlp_d512.yaml"
    "04_mlp_d${DIM_ALIGNMENT}_struct      configs/mlp/vits_minilm/mlp_d512_struct.yaml"
    "05_csa_d256                          configs/csa/vits_minilm/csa_d256.yaml"
    "06_csa_d256_struct                   configs/csa/vits_minilm/csa_d256_struct.yaml"
)

COMPLETED=()
for entry in "${CONFIGS[@]}"; do
    read -r NAME CFG <<< "$entry"
    if [[ ! -f "$CFG" ]]; then
        echo "[SKIP] $NAME — missing config $CFG" | tee -a "$LOG"
        continue
    fi
    echo "[$(date)] >>> $NAME ($CFG)" | tee -a "$LOG"
    PYTHONPATH=. python src/train_alignment.py --config_path "$CFG" 2>&1 | tee -a "$LOG"
    COMPLETED+=("$NAME")
done

echo "[$(date)] Done. Completed: ${COMPLETED[*]:-none}. Wall: $(( $(date +%s) - START ))s" | tee -a "$LOG"
