#!/bin/bash
# Re-evaluate all methods on COCO Karpathy 5K split for retrieval.
set -u
export CUDA_VISIBLE_DEVICES="${1:-0}"
export PYTHONPATH="${PYTHONPATH:-.}"
export MALLOC_ARENA_MAX=2
cd /workspace/STRUCTURE

RDIR="results/alignment-sentence_transformers_all_roberta_large_v1-vit_large_patch14_dinov2.lvd142m"
mkdir -p logs/karpathy_reeval

RUNS=(
    "cls_linear|configs/linear/vitl_roberta/linear_d512.yaml|cool-universe-39"
    "cls_linear_str|configs/linear/vitl_roberta/linear_d512_struct.yaml|ethereal-sunset-40"
    "cls_mlp|configs/mlp/vitl_roberta/mlp_d512.yaml|rose-terrain-41"
    "cls_mlp_str|configs/mlp/vitl_roberta/mlp_d512_struct.yaml|hardy-haze-42"
    "cls_sail|configs/sail/vitl_roberta/sail_star_cls.yaml|giddy-meadow-112"
    "cls_sail_str|configs/sail/vitl_roberta/sail_star_cls_struct.yaml|twilight-darkness-114"
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

echo "[$(date)] Karpathy 5K retrieval reeval: ${#RUNS[@]} checkpoints on GPU $CUDA_VISIBLE_DEVICES"

for row in "${RUNS[@]}"; do
    IFS='|' read -r label cfg run <<< "$row"
    ckpt=$(ls "${RDIR}-${run}"/'(23, 24)_0.2903'/checkpoints/checkpoint-*.pth 2>/dev/null | head -1)
    if [ -z "$ckpt" ]; then
        echo "[SKIP] $label — no checkpoint"
        continue
    fi
    log="logs/karpathy_reeval/${label}.log"
    echo "[$(date '+%H:%M:%S')] START $label"
    python -u rerun_eval.py \
        --config_path "$cfg" \
        --ckpt "$ckpt" \
        --label "$label" \
        --rt coco_karpathy \
        --img_layer 23 --txt_layer 24 \
        > "$log" 2>&1
    echo "[$(date '+%H:%M:%S')] DONE  $label  exit=$?"
done

echo "[$(date)] ALL DONE"
