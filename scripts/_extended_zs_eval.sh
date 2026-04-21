#!/bin/bash
# Extended zero-shot classification eval on all available datasets
# for all vitl_roberta checkpoints. Runs on GPU 1 while training on GPU 0.
set -u
export CUDA_VISIBLE_DEVICES="${1:-1}"
export PYTHONPATH="${PYTHONPATH:-.}"
export MALLOC_ARENA_MAX=2
cd /workspace/STRUCTURE

ZS_DATASETS="stl10,caltech101,food101,cifar10,cifar100,imagenet,sun397,eurosat,mnist,dtd,gtsrb,country211,cars,aircraft,pets,flowers"
RDIR="results/alignment-sentence_transformers_all_roberta_large_v1-vit_large_patch14_dinov2.lvd142m"

mkdir -p logs/extended_zs_eval

# All checkpoints: label|config|run_name
RUNS=(
    # CLS methods
    "cls_linear|configs/linear/vitl_roberta/linear_d512.yaml|cool-universe-39"
    "cls_linear_str|configs/linear/vitl_roberta/linear_d512_struct.yaml|ethereal-sunset-40"
    "cls_mlp|configs/mlp/vitl_roberta/mlp_d512.yaml|rose-terrain-41"
    "cls_mlp_str|configs/mlp/vitl_roberta/mlp_d512_struct.yaml|hardy-haze-42"
    "cls_sail|configs/sail/vitl_roberta/sail_star_cls.yaml|giddy-meadow-112"
    "cls_sail_str|configs/sail/vitl_roberta/sail_star_cls_struct.yaml|twilight-darkness-114"
    "cls_ba_k32|configs/ba/vitl_roberta/cls_k32.yaml|fallen-serenity-111"
    "cls_ba_k64|configs/ba/vitl_roberta/cls_k64.yaml|sandy-feather-113"
    "cls_ba_k128|configs/ba/vitl_roberta/cls_k128.yaml|eager-capybara-43"
    "cls_ba_k256|configs/ba/vitl_roberta/cls_k256.yaml|toasty-vortex-30"
    "cls_ba_k512|configs/ba/vitl_roberta/cls_k512.yaml|scarlet-galaxy-31"
    # Token methods
    "token_linear|configs/linear/vitl_roberta/linear_d512_token.yaml|confused-snowflake-55"
    "token_linear_str|configs/linear/vitl_roberta/linear_d512_token_struct.yaml|fresh-energy-60"
    "token_mlp|configs/mlp/vitl_roberta/mlp_d512_token.yaml|fresh-firebrand-61"
    "token_mlp_str|configs/mlp/vitl_roberta/mlp_d512_token_struct.yaml|lucky-snow-84"
    "token_fa|configs/freezealign/vitl_roberta/fa_d512.yaml|visionary-yogurt-51"
    "token_fa_str|configs/freezealign/vitl_roberta/fa_d512_struct.yaml|soft-voice-92"
    "token_ba_k128|configs/ba/vitl_roberta/token_k128.yaml|dutiful-fire-49"
    "token_ba_k256|configs/ba/vitl_roberta/token_k256.yaml|balmy-pond-50"
    "token_ba_k512|configs/ba/vitl_roberta/token_k512.yaml|generous-elevator-48"
    "token_ba_k1024|configs/ba/vitl_roberta/token_k1024.yaml|glamorous-disco-105"
)

echo "[$(date)] Extended ZS eval: ${#RUNS[@]} checkpoints × 16 datasets on GPU $CUDA_VISIBLE_DEVICES"

for row in "${RUNS[@]}"; do
    IFS='|' read -r label cfg run <<< "$row"
    ckpt=$(ls "${RDIR}-${run}"/'(23, 24)_0.2903'/checkpoints/checkpoint-*.pth 2>/dev/null | head -1)
    if [ -z "$ckpt" ]; then
        echo "[SKIP] $label — no checkpoint"
        continue
    fi
    log="logs/extended_zs_eval/${label}.log"
    echo "[$(date '+%H:%M:%S')] START $label"
    python -u rerun_eval.py \
        --config_path "$cfg" \
        --ckpt "$ckpt" \
        --label "$label" \
        --zs "$ZS_DATASETS" \
        --img_layer 23 --txt_layer 24 \
        > "$log" 2>&1
    echo "[$(date '+%H:%M:%S')] DONE  $label  exit=$?"
done

echo "[$(date)] ALL DONE (extended ZS eval)"
