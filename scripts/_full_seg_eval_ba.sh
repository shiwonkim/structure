#!/bin/bash
# Full segmentation eval for all Token BA K values on all seg datasets.
# Re-eval needed: code changed (softmax added to anchor_codebook).
set -u
export CUDA_VISIBLE_DEVICES="${1:-0}"
export PYTHONPATH="${PYTHONPATH:-.}"
export MALLOC_ARENA_MAX=2
cd /workspace/STRUCTURE

RDIR="results/alignment-sentence_transformers_all_roberta_large_v1-vit_large_patch14_dinov2.lvd142m"
BA_METHODS="direct_cosine,anchor_codebook,attention_map"
STRATS="raw,ensemble"

mkdir -p logs/seg_eval_v2

RUNS=(
    "token_ba_k64|configs/ba/vitl_roberta/token_k64.yaml|balmy-morning-121"
    "token_ba_k128|configs/ba/vitl_roberta/token_k128.yaml|dutiful-fire-49"
    "token_ba_k256|configs/ba/vitl_roberta/token_k256.yaml|balmy-pond-50"
    "token_ba_k512|configs/ba/vitl_roberta/token_k512.yaml|generous-elevator-48"
    "token_ba_k1024|configs/ba/vitl_roberta/token_k1024.yaml|glamorous-disco-105"
)

DATASETS=(
    "voc2012|data/pascal_voc|"
    "ade20k|data/ade20k|"
    "pascal_context|data/pascal_context|--max-images 2000"
    "coco_object|data/coco_seg|"
    "coco_stuff|data/coco_seg|"
)

echo "[$(date)] Full seg eval: ${#RUNS[@]} checkpoints × ${#DATASETS[@]} datasets on GPU $CUDA_VISIBLE_DEVICES"

for row in "${RUNS[@]}"; do
    IFS='|' read -r label cfg run <<< "$row"
    ckpt=$(ls "${RDIR}-${run}"/'(23, 24)_0.2903'/checkpoints/checkpoint-*.pth 2>/dev/null | head -1)
    if [ -z "$ckpt" ]; then
        echo "[SKIP] $label — no checkpoint"
        continue
    fi

    for ds_row in "${DATASETS[@]}"; do
        IFS='|' read -r ds root extra <<< "$ds_row"
        log="logs/seg_eval_v2/${label}_${ds}.log"
        csv="logs/seg_eval_v2/${label}_${ds}.csv"
        echo "[$(date '+%H:%M:%S')] START $label × $ds"
        python -u src/evaluation/zero_shot_segmentation.py \
            --config "$cfg" \
            --checkpoint "$ckpt" \
            --layer-img 23 --layer-txt 24 \
            --dataset "$ds" --data-root "$root" \
            --methods "$BA_METHODS" \
            --text-strategies "$STRATS" \
            --gpu 0 \
            --output-csv "$csv" \
            $extra \
            > "$log" 2>&1
        echo "[$(date '+%H:%M:%S')] DONE  $label × $ds  exit=$?"
    done
done

echo "[$(date)] ALL DONE"
