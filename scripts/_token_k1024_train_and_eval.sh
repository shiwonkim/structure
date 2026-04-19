#!/bin/bash
# Token BA K=1024 on ViT-L + RoBERTa-Large.
# Training + retrieval/classification (built into trainer) + segmentation eval.
set -euo pipefail
cd /workspace/STRUCTURE
export MALLOC_ARENA_MAX=2
export PYTHONPATH="${PYTHONPATH:-.}"

GPU=${1:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M)
CONFIG="configs/ba/vitl_roberta/token_k1024.yaml"

echo "[$(date)] Token BA K=1024 training on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU python src/train_alignment.py \
    --config_path "$CONFIG" \
    2>&1 | tee logs/vitl_roberta_token_k1024_${TIMESTAMP}.log

echo "[$(date)] Training done. Finding checkpoint..."
RDIR=$(ls -td results/alignment-sentence_transformers_all_roberta_large_v1-vit_large_patch14_dinov2.lvd142m-*/ 2>/dev/null | head -1)
CKPT=$(ls "$RDIR"/'(23, 24)_0.2903'/checkpoints/checkpoint-*.pth 2>/dev/null | head -1)

if [ -z "$CKPT" ]; then
    echo "[$(date)] ERROR: No checkpoint found in $RDIR"
    exit 1
fi
echo "[$(date)] Checkpoint: $CKPT"

echo "[$(date)] Running segmentation eval on GPU $GPU"
BA_METHODS="direct_cosine,anchor_codebook,attention_map"
STRATS="raw,ensemble"
mkdir -p logs/seg_eval

for ds_row in "voc2012|data/pascal_voc|" "ade20k|data/ade20k|" "cityscapes|data/cityscapes|" "pascal_context|data/pascal_context|--max-images 2000"; do
    IFS='|' read -r ds root extra <<< "$ds_row"
    log="logs/seg_eval/token_ba_k1024_${ds}.log"
    csv="logs/seg_eval/token_ba_k1024_${ds}.csv"
    echo "[$(date)] Seg eval: $ds"
    CUDA_VISIBLE_DEVICES=$GPU python -u src/evaluation/zero_shot_segmentation.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --layer-img 23 --layer-txt 24 \
        --dataset "$ds" --data-root "$root" \
        --methods "$BA_METHODS" \
        --text-strategies "$STRATS" \
        --gpu 0 \
        --output-csv "$csv" \
        $extra \
        > "$log" 2>&1
    echo "[$(date)] Seg eval $ds done (exit=$?)"
done

echo "[$(date)] All done — Token BA K=1024 training + eval complete."
