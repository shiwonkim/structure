#!/bin/bash
# ZS eval for main table methods on all 15 available datasets.
# Covers: missing SAIL concat full eval + 5 new datasets for all methods.
set -u
export CUDA_VISIBLE_DEVICES="${1:-1}"
export PYTHONPATH="${PYTHONPATH:-.}"
export MALLOC_ARENA_MAX=2
cd /workspace/STRUCTURE

RDIR="results/alignment-sentence_transformers_all_roberta_large_v1-vit_large_patch14_dinov2.lvd142m"
ALL_DS="stl10,caltech101,food101,cifar10,cifar100,imagenet,eurosat,mnist,dtd,gtsrb,country211,pets,aircraft,flowers,sun397"
NEW_DS="pets,aircraft,eurosat,caltech101,sun397"

mkdir -p logs/main_table_zs_eval

# SAIL concat: full 15 datasets (no prior eval)
SAIL_RUNS=(
    "sail_concat|configs/sail/vitl_roberta/sail_star_concat.yaml|summer-breeze-124"
    "sail_concat_str|configs/sail/vitl_roberta/sail_star_concat_struct.yaml|rare-pine-125"
)

# Other methods: only 5 new datasets (already have 10)
OTHER_RUNS=(
    "cls_linear_str|configs/linear/vitl_roberta/linear_d512_struct.yaml|ethereal-sunset-40"
    "cls_mlp_str|configs/mlp/vitl_roberta/mlp_d512_struct.yaml|hardy-haze-42"
    "token_fa|configs/freezealign/vitl_roberta/fa_d512.yaml|visionary-yogurt-51"
    "token_fa_str|configs/freezealign/vitl_roberta/fa_d512_struct.yaml|soft-voice-92"
    "token_ba_k512|configs/ba/vitl_roberta/token_k512.yaml|generous-elevator-48"
)

echo "[$(date)] Main table ZS eval on GPU $CUDA_VISIBLE_DEVICES"

# SAIL concat: retrain first (no checkpoints from crashed runs)
echo "[$(date)] Retraining SAIL concat on GPU $CUDA_VISIBLE_DEVICES"
echo "[$(date)] >>> sail_concat"
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python src/train_alignment.py \
    --config_path configs/sail/vitl_roberta/sail_star_concat.yaml 2>&1 | \
    tee logs/main_table_zs_eval/sail_concat_train.log
echo "[$(date)] <<< sail_concat done"

echo "[$(date)] >>> sail_concat_struct"
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python src/train_alignment.py \
    --config_path configs/sail/vitl_roberta/sail_star_concat_struct.yaml 2>&1 | \
    tee logs/main_table_zs_eval/sail_concat_str_train.log
echo "[$(date)] <<< sail_concat_struct done"

# Now run ZS eval on 5 new datasets for SAIL concat (training eval already covers the original set)
# Find the latest SAIL concat checkpoints
for row in "${SAIL_RUNS[@]}"; do
    IFS='|' read -r label cfg run <<< "$row"
    # Find newest matching run dir
    ckpt=$(find "${RDIR}-"* -path "*checkpoints/checkpoint-*.pth" -newer logs/main_table_zs_eval/sail_concat_train.log 2>/dev/null | head -1)
    if [ -z "$ckpt" ]; then
        echo "[SKIP] $label — no checkpoint after retrain"
        continue
    fi
    log="logs/main_table_zs_eval/${label}_new5.log"
    echo "[$(date '+%H:%M:%S')] START $label (5 new datasets)"
    python -u rerun_eval.py \
        --config_path "$cfg" \
        --ckpt "$ckpt" \
        --label "$label" \
        --zs "$NEW_DS" \
        --img_layer 23 --txt_layer 24 \
        > "$log" 2>&1
    echo "[$(date '+%H:%M:%S')] DONE  $label  exit=$?"
done

# Other methods: only 5 new datasets
for row in "${OTHER_RUNS[@]}"; do
    IFS='|' read -r label cfg run <<< "$row"
    ckpt=$(ls "${RDIR}-${run}"/'(23, 24)_0.2903'/checkpoints/checkpoint-*.pth 2>/dev/null | head -1)
    if [ -z "$ckpt" ]; then
        echo "[SKIP] $label — no checkpoint"
        continue
    fi
    log="logs/main_table_zs_eval/${label}_new5.log"
    echo "[$(date '+%H:%M:%S')] START $label (5 new datasets)"
    python -u rerun_eval.py \
        --config_path "$cfg" \
        --ckpt "$ckpt" \
        --label "$label" \
        --zs "$NEW_DS" \
        --img_layer 23 --txt_layer 24 \
        > "$log" 2>&1
    echo "[$(date '+%H:%M:%S')] DONE  $label  exit=$?"
done

echo "[$(date)] ALL DONE"
