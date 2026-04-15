#!/bin/bash
# Zero-shot segmentation eval launcher.
#
# Runs src/evaluation/zero_shot_segmentation.py against a trained checkpoint,
# producing a mIoU table for all compatible (method, strategy) pairs on
# Pascal VOC 2012 val. Supports BA, FreezeAlign, and no-checkpoint baselines.
#
# Usage:
#   bash scripts/eval_segmentation.sh <config> <checkpoint> [gpu] [layer_img] [layer_txt]
#
# Example:
#   bash scripts/eval_segmentation.sh \
#       configs/ba/vits_minilm/token_k256.yaml \
#       results/alignment-.../checkpoint-epoch400.pth \
#       0 11 6
#
# For the no-checkpoint (direct_cosine only) mode, pass the empty string as
# checkpoint:
#   bash scripts/eval_segmentation.sh configs/ba/vits_minilm/token_k256.yaml "" 0 11 6
set -euo pipefail

CFG=${1:?missing config path}
CKPT=${2:-}
GPU=${3:-0}
LAYER_IMG=${4:-11}
LAYER_TXT=${5:-6}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

mkdir -p logs
LOG="logs/seg_eval_$(date +%Y%m%d_%H%M).log"

CMD=(PYTHONPATH=. python src/evaluation/zero_shot_segmentation.py
     --config "$CFG"
     --layer-img "$LAYER_IMG"
     --layer-txt "$LAYER_TXT"
     --dataset voc2012
     --data-root data/pascal_voc
     --download
     --methods "direct_cosine,freezealign,anchor_codebook,attention_map"
     --text-strategies "raw,ensemble"
     --gpu "$GPU")

if [[ -n "$CKPT" ]]; then
    CMD+=(--checkpoint "$CKPT")
fi

echo "[$(date)] $ ${CMD[*]}" | tee "$LOG"
"${CMD[@]}" 2>&1 | tee -a "$LOG"
