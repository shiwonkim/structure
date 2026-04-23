#!/usr/bin/env bash
set -euo pipefail
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate structure

GPU="${1:-1}"
CKPT='results/alignment-sentence_transformers_all_mpnet_base_v2-vit_base_patch14_dinov2.lvd142m-rich-flower-83/(11, 12)_0.2852/checkpoints/checkpoint-epoch408.pth'
CONFIG="configs/ba/vitb_mpnet/token_k512.yaml"
LOGDIR="logs/seg_vitb_k512_fixed"
mkdir -p "$LOGDIR"

DATASETS=("voc2012" "pascal_context" "ade20k" "coco_object" "coco_stuff")
DATA_ROOTS=("data/pascal_voc" "data/pascal_context" "data/ade20k" "data/coco_seg" "data/coco_seg")
MAX_IMAGES=("" "2000" "2000" "" "")

for i in "${!DATASETS[@]}"; do
    ds="${DATASETS[$i]}"
    dr="${DATA_ROOTS[$i]}"
    mi="${MAX_IMAGES[$i]}"
    extra=""
    if [ -n "$mi" ]; then extra="--max-images $mi"; fi
    echo "[$(date +%H:%M:%S)] START $ds"
    PYTHONPATH=. python src/evaluation/zero_shot_segmentation.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --layer-img 11 --layer-txt 12 \
        --dataset "$ds" --data-root "$dr" \
        --methods "anchor_codebook" \
        --text-strategies ensemble \
        --gpu "$GPU" $extra \
        2>&1 | tee "$LOGDIR/${ds}.log"
    echo "[$(date +%H:%M:%S)] DONE  $ds"
done
echo "[$(date +%H:%M:%S)] ALL DONE"
