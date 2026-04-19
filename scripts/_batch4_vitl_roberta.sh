#!/bin/bash
# Batch 4: token-level Linear/MLP + FreezeAlign+STR rerun
# Sequential on GPU 0 (CommitLimit prevents concurrent runs)
set -e
cd /workspace/STRUCTURE
export MALLOC_ARENA_MAX=2
GPU=${1:-0}
LOGDIR=logs
TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG="$LOGDIR/vitl_roberta_batch4_${TIMESTAMP}.log"

CONFIGS=(
    "01_linear_token:configs/linear/vitl_roberta/linear_d512_token.yaml"
    "02_linear_token_struct:configs/linear/vitl_roberta/linear_d512_token_struct.yaml"
    "03_mlp_token:configs/mlp/vitl_roberta/mlp_d512_token.yaml"
    "04_mlp_token_struct:configs/mlp/vitl_roberta/mlp_d512_token_struct.yaml"
    "05_freezealign_struct:configs/freezealign/vitl_roberta/fa_d512_struct.yaml"
)

echo "[$(date)] Batch 4: vitl_roberta on GPU $GPU" | tee -a "$LOG"

for entry in "${CONFIGS[@]}"; do
    name="${entry%%:*}"
    config="${entry##*:}"
    echo "[$(date)] >>> $name ($config)" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. python src/train_alignment.py \
        --config_path "$config" 2>&1 | tee -a "$LOG"
    echo "[$(date)] <<< $name done" | tee -a "$LOG"
done

echo "[$(date)] Batch 4 complete." | tee -a "$LOG"
