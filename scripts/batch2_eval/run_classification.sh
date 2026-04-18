#!/usr/bin/env bash
# Re-run zero-shot classification + retrieval eval for every Batch 2 checkpoint.
# Writes one log per (run, mode) under logs/batch2_eval/.
# Usage: bash scripts/batch2_eval/run_classification.sh <gpu_id>

set -u
GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="${PYTHONPATH:-.}"

cd "$(dirname "$0")/../.."
mkdir -p logs/batch2_eval configs/_tmp_eval

PY=/home/shiwon/miniconda3/envs/structure/bin/python
ZS="cifar10,cifar100,stl10,mnist,dtd,flowers,gtsrb,country211"
RT="coco,flickr30"
RDIR="results/alignment-sentence_transformers_all_MiniLM_L6_v2-vit_small_patch14_dinov2.lvd142m"

# Fields separated by '|'. Last field = 1 if token method (needs 2nd CLS-fallback pass).
RUNS=(
  "01_linear|configs/linear/vits_minilm/linear_d512.yaml|neat-puddle-3|(11, 6)_nan/checkpoints/checkpoint-epoch484.pth|0"
  "02_linear_struct|configs/linear/vits_minilm/linear_d512_struct.yaml|super-sponge-7|(11, 6)_0.2739/checkpoints/checkpoint-epoch480.pth|0"
  "03_mlp|configs/mlp/vits_minilm/mlp_d512.yaml|swift-cosmos-9|(11, 6)_0.2739/checkpoints/checkpoint-epoch479.pth|0"
  "04_mlp_struct|configs/mlp/vits_minilm/mlp_d512_struct.yaml|easy-wood-11|(11, 6)_0.2739/checkpoints/checkpoint-epoch479.pth|0"
  "05_cls_ba_k128|configs/ba/vits_minilm/cls_k128.yaml|dulcet-donkey-5|(11, 6)_nan/checkpoints/checkpoint-epoch513.pth|0"
  "06_cls_ba_k256|configs/ba/vits_minilm/cls_k256.yaml|iconic-wave-6|(11, 6)_0.2739/checkpoints/checkpoint-epoch598.pth|0"
  "07_cls_ba_k512|configs/ba/vits_minilm/cls_k512.yaml|true-microwave-8|(11, 6)_0.2739/checkpoints/checkpoint-epoch505.pth|0"
  "08_token_ba_k128|configs/ba/vits_minilm/token_k128.yaml|icy-vortex-10|(11, 6)_0.2739/checkpoints/checkpoint-epoch588.pth|1"
  "09_token_ba_k256|configs/ba/vits_minilm/token_k256.yaml|flowing-resonance-13|(11, 6)_0.2739/checkpoints/checkpoint-epoch411.pth|1"
  "10_token_ba_k512|configs/ba/vits_minilm/token_k512.yaml|wobbly-water-15|(11, 6)_0.2739/checkpoints/checkpoint-epoch490.pth|1"
  "11_fa_d512|configs/freezealign/vits_minilm/fa_d512.yaml|glowing-lake-14|(11, 6)_0.2739/checkpoints/checkpoint-epoch551.pth|1"
  "12_fa_struct|configs/freezealign/vits_minilm/fa_d512_struct.yaml|faithful-salad-17|(11, 6)_0.2739/checkpoints/checkpoint-epoch999.pth|1"
  "13_token_ba_k128_p|configs/ba/vits_minilm/token_k128_prior.yaml|bright-resonance-21|(11, 6)_0.2739/checkpoints/checkpoint-epoch999.pth|1"
  "14_token_ba_k256_p|configs/ba/vits_minilm/token_k256_prior.yaml|feasible-wildflower-28|(11, 6)_0.2739/checkpoints/checkpoint-epoch499.pth|1"
  "15_token_ba_k512_p|configs/ba/vits_minilm/token_k512_prior.yaml|polished-feather-35|(11, 6)_0.2739/checkpoints/checkpoint-epoch490.pth|1"
)

for row in "${RUNS[@]}"; do
  IFS='|' read -r label cfg run ckpt_rel need_cls_pass <<< "$row"
  ckpt="${RDIR}-${run}/${ckpt_rel}"

  if [ ! -f "$ckpt" ]; then
    echo "[MISS] $label  $ckpt" >&2
    continue
  fi

  # Pass 1: native mode (config-default).
  log="logs/batch2_eval/${label}_native.log"
  echo "[$(date '+%H:%M:%S')] START $label  native"
  $PY -u rerun_eval.py \
    --config_path "$cfg" \
    --ckpt "$ckpt" \
    --label "${label}_native" \
    --zs "$ZS" \
    --rt "$RT" \
    --img_layer 11 --txt_layer 6 \
    > "$log" 2>&1
  echo "[$(date '+%H:%M:%S')] DONE  $label  native  exit=$?"

  # Pass 2: CLS-fallback mode for token methods.
  if [ "$need_cls_pass" = "1" ]; then
    tmp_cfg="configs/_tmp_eval/${label}_cls.yaml"
    PYTHONPATH=. $PY -c "
import yaml
from src.core.src.utils.loader import Loader
with open('$cfg') as f:
    data = yaml.load(f, Loader=Loader)
ov = data.setdefault('overrides', {})
ev = ov.setdefault('evaluation', {})
ev['token_level_zero_shot'] = False
with open('$tmp_cfg', 'w') as f:
    yaml.safe_dump(data, f)
"
    log="logs/batch2_eval/${label}_clsfallback.log"
    echo "[$(date '+%H:%M:%S')] START $label  clsfallback"
    $PY -u rerun_eval.py \
      --config_path "$tmp_cfg" \
      --ckpt "$ckpt" \
      --label "${label}_clsfallback" \
      --zs "$ZS" \
      --rt "$RT" \
      --img_layer 11 --txt_layer 6 \
      > "$log" 2>&1
    echo "[$(date '+%H:%M:%S')] DONE  $label  clsfallback  exit=$?"
  fi
done

echo "[$(date '+%H:%M:%S')] ALL DONE (classification+retrieval on GPU $GPU)"
