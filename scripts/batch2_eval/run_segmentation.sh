#!/usr/bin/env bash
# Zero-shot segmentation eval for Token BA + FreezeAlign Batch 2 checkpoints.
# 4 datasets × 2 text strategies × (dispatched) methods.
# Usage: bash scripts/batch2_eval/run_segmentation.sh <gpu_id>

set -u
GPU="${1:-1}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="${PYTHONPATH:-.}"

cd "$(dirname "$0")/../.."
mkdir -p logs/batch2_eval

PY=/home/shiwon/miniconda3/envs/structure/bin/python
RDIR="results/alignment-sentence_transformers_all_MiniLM_L6_v2-vit_small_patch14_dinov2.lvd142m"

METHODS="direct_cosine,freezealign,anchor_codebook,attention_map"
STRATS="raw,ensemble"

# 8 Token-BA + FreezeAlign checkpoints. Fields: label | config | run | ckpt_rel
RUNS=(
  "seg_token_ba_k128|configs/ba/vits_minilm/token_k128.yaml|icy-vortex-10|(11, 6)_0.2739/checkpoints/checkpoint-epoch588.pth"
  "seg_token_ba_k256|configs/ba/vits_minilm/token_k256.yaml|flowing-resonance-13|(11, 6)_0.2739/checkpoints/checkpoint-epoch411.pth"
  "seg_token_ba_k512|configs/ba/vits_minilm/token_k512.yaml|wobbly-water-15|(11, 6)_0.2739/checkpoints/checkpoint-epoch490.pth"
  "seg_fa_d512|configs/freezealign/vits_minilm/fa_d512.yaml|glowing-lake-14|(11, 6)_0.2739/checkpoints/checkpoint-epoch551.pth"
  "seg_fa_struct|configs/freezealign/vits_minilm/fa_d512_struct.yaml|faithful-salad-17|(11, 6)_0.2739/checkpoints/checkpoint-epoch999.pth"
  "seg_token_ba_k128_p|configs/ba/vits_minilm/token_k128_prior.yaml|bright-resonance-21|(11, 6)_0.2739/checkpoints/checkpoint-epoch999.pth"
  "seg_token_ba_k256_p|configs/ba/vits_minilm/token_k256_prior.yaml|feasible-wildflower-28|(11, 6)_0.2739/checkpoints/checkpoint-epoch499.pth"
  "seg_token_ba_k512_p|configs/ba/vits_minilm/token_k512_prior.yaml|polished-feather-35|(11, 6)_0.2739/checkpoints/checkpoint-epoch490.pth"
)

# Dataset | data-root | extra args (e.g. --max-images for pascal_context)
DATASETS=(
  "voc2012|data/pascal_voc|"
  "ade20k|data/ade20k|"
  "cityscapes|data/cityscapes|"
  "pascal_context|data/pascal_context|--max-images 2000"
)

for row in "${RUNS[@]}"; do
  IFS='|' read -r label cfg run ckpt_rel <<< "$row"
  ckpt="${RDIR}-${run}/${ckpt_rel}"
  if [ ! -f "$ckpt" ]; then
    echo "[MISS] $label  $ckpt" >&2
    continue
  fi

  for ds_row in "${DATASETS[@]}"; do
    IFS='|' read -r ds root extra <<< "$ds_row"
    log="logs/batch2_eval/${label}_${ds}.log"
    echo "[$(date '+%H:%M:%S')] START $label  $ds"
    $PY -u src/evaluation/zero_shot_segmentation.py \
      --config "$cfg" \
      --checkpoint "$ckpt" \
      --layer-img 11 --layer-txt 6 \
      --dataset "$ds" --data-root "$root" \
      --methods "$METHODS" \
      --text-strategies "$STRATS" \
      --gpu 0 \
      $extra \
      > "$log" 2>&1
    ec=$?
    echo "[$(date '+%H:%M:%S')] DONE  $label  $ds  exit=$ec"
  done
done

echo "[$(date '+%H:%M:%S')] ALL DONE (segmentation on GPU $GPU)"
