#!/bin/bash
# ZS eval on 5 new datasets for methods that already have checkpoints.
set -u
export CUDA_VISIBLE_DEVICES="${1:-1}"
export PYTHONPATH="${PYTHONPATH:-.}"
export MALLOC_ARENA_MAX=2
cd /workspace/STRUCTURE

RDIR="results/alignment-sentence_transformers_all_roberta_large_v1-vit_large_patch14_dinov2.lvd142m"
NEW_DS="pets,aircraft,eurosat,caltech101,sun397"

mkdir -p logs/main_table_zs_eval

RUNS=(
    "cls_linear_str|configs/linear/vitl_roberta/linear_d512_struct.yaml|ethereal-sunset-40"
    "cls_mlp_str|configs/mlp/vitl_roberta/mlp_d512_struct.yaml|hardy-haze-42"
    "token_fa|configs/freezealign/vitl_roberta/fa_d512.yaml|visionary-yogurt-51"
    "token_fa_str|configs/freezealign/vitl_roberta/fa_d512_struct.yaml|soft-voice-92"
    "token_ba_k512|configs/ba/vitl_roberta/token_k512.yaml|generous-elevator-48"
)

echo "[$(date)] ZS eval on 5 new datasets, GPU $CUDA_VISIBLE_DEVICES"

for row in "${RUNS[@]}"; do
    IFS='|' read -r label cfg run <<< "$row"
    ckpt=$(ls "${RDIR}-${run}"/'(23, 24)_0.2903'/checkpoints/checkpoint-*.pth 2>/dev/null | head -1)
    if [ -z "$ckpt" ]; then
        echo "[SKIP] $label — no checkpoint"
        continue
    fi
    log="logs/main_table_zs_eval/${label}_new5.log"
    echo "[$(date '+%H:%M:%S')] START $label"
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
