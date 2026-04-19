#!/usr/bin/env bash
# Zero-shot segmentation eval for all token-level vitl_roberta runs.
# 6 checkpoints × 4 datasets × compatible methods.
# Runs on GPU 1 while Batch 4 training continues on GPU 0.
set -u
GPU="${1:-1}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="${PYTHONPATH:-.}"
export MALLOC_ARENA_MAX=2

cd "$(dirname "$0")/.."
mkdir -p logs/seg_eval

RDIR="results/alignment-sentence_transformers_all_roberta_large_v1-vit_large_patch14_dinov2.lvd142m"

# Methods per checkpoint type:
#   BA checkpoints:       direct_cosine,anchor_codebook,attention_map
#   FreezeAlign:          direct_cosine,freezealign
#   Linear/MLP (token):   direct_cosine,linear_perpatch
BA_METHODS="direct_cosine,anchor_codebook,attention_map"
FA_METHODS="direct_cosine,freezealign"
LIN_METHODS="direct_cosine,linear_perpatch"
STRATS="raw,ensemble"

# label | config | run_name | ckpt_rel | methods
RUNS=(
  "token_ba_k128|configs/ba/vitl_roberta/token_k128.yaml|dutiful-fire-49|(23, 24)_0.2903/checkpoints/checkpoint-epoch296.pth|$BA_METHODS"
  "token_ba_k256|configs/ba/vitl_roberta/token_k256.yaml|balmy-pond-50|(23, 24)_0.2903/checkpoints/checkpoint-epoch404.pth|$BA_METHODS"
  "token_ba_k512|configs/ba/vitl_roberta/token_k512.yaml|generous-elevator-48|(23, 24)_0.2903/checkpoints/checkpoint-epoch400.pth|$BA_METHODS"
  "freezealign|configs/freezealign/vitl_roberta/fa_d512.yaml|visionary-yogurt-51|(23, 24)_0.2903/checkpoints/checkpoint-epoch205.pth|$FA_METHODS"
  "linear_token|configs/linear/vitl_roberta/linear_d512_token.yaml|confused-snowflake-55|(23, 24)_0.2903/checkpoints/checkpoint-epoch380.pth|$LIN_METHODS"
  "linear_token_str|configs/linear/vitl_roberta/linear_d512_token_struct.yaml|fresh-energy-60|(23, 24)_0.2903/checkpoints/checkpoint-epoch667.pth|$LIN_METHODS"
)

DATASETS=(
  "voc2012|data/pascal_voc|"
  "ade20k|data/ade20k|"
  "cityscapes|data/cityscapes|"
  "pascal_context|data/pascal_context|--max-images 2000"
)

echo "[$(date)] Segmentation eval: ${#RUNS[@]} runs × ${#DATASETS[@]} datasets on GPU $GPU"

for row in "${RUNS[@]}"; do
  IFS='|' read -r label cfg run ckpt_rel methods <<< "$row"
  ckpt="${RDIR}-${run}/${ckpt_rel}"
  if [ ! -f "$ckpt" ]; then
    echo "[MISS] $label  $ckpt" >&2
    continue
  fi

  for ds_row in "${DATASETS[@]}"; do
    IFS='|' read -r ds root extra <<< "$ds_row"
    log="logs/seg_eval/${label}_${ds}.log"
    csv="logs/seg_eval/${label}_${ds}.csv"
    echo "[$(date '+%H:%M:%S')] START $label  $ds  methods=$methods"
    python -u src/evaluation/zero_shot_segmentation.py \
      --config "$cfg" \
      --checkpoint "$ckpt" \
      --layer-img 23 --layer-txt 24 \
      --dataset "$ds" --data-root "$root" \
      --methods "$methods" \
      --text-strategies "$STRATS" \
      --gpu 0 \
      --output-csv "$csv" \
      $extra \
      > "$log" 2>&1
    ec=$?
    echo "[$(date '+%H:%M:%S')] DONE  $label  $ds  exit=$ec"
  done
done

echo "[$(date)] ALL DONE (segmentation eval on GPU $GPU)"
