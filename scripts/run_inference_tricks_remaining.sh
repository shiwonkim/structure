#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate structure

GPU="${1:-1}"
CKPT='results/alignment-sentence_transformers_all_MiniLM_L6_v2-vit_small_patch14_dinov2.lvd142m-wobbly-water-15/(11, 6)_0.2739/checkpoints/checkpoint-epoch490.pth'
CONFIG="configs/ba/vits_minilm/token_k512.yaml"
LOGDIR="logs/inference_tricks"
mkdir -p "$LOGDIR"

DATASETS=("pascal_context" "ade20k")
DATA_ROOTS=("data/pascal_context" "data/ade20k/ADEChallengeData2016")
MAX_IMAGES=("2000" "2000")

METHODS="anchor_codebook,iterative_cap_2,iterative_cap_3,iterative_cap_5,tau_sharp_0.05,tau_sharp_0.03,tau_sharp_0.01,tau_sharp_0.005"

for i in "${!DATASETS[@]}"; do
    ds="${DATASETS[$i]}"
    dr="${DATA_ROOTS[$i]}"
    mi="${MAX_IMAGES[$i]}"
    LOGFILE="$LOGDIR/tricks_${ds}.log"

    echo "[$(date +%H:%M:%S)] START $ds"
    PYTHONPATH=. python src/evaluation/zero_shot_segmentation.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --layer-img 11 --layer-txt 6 \
        --dataset "$ds" \
        --data-root "$dr" \
        --methods "$METHODS" \
        --text-strategies ensemble \
        --gpu "$GPU" \
        --max-images $mi \
        2>&1 | tee "$LOGFILE"

    echo "[$(date +%H:%M:%S)] DONE  $ds"
done

echo "[$(date +%H:%M:%S)] ALL DONE"
