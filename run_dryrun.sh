#!/bin/bash
# Dry-run wrapper: disable cuDNN due to driver 470 / cu118 incompatibility
export WANDB_MODE=offline
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=1

python -c "
import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
# Now run the actual training
import sys, runpy
sys.argv = ['src/train_alignment.py', '--config_path', 'configs/dryrun.yaml']
runpy.run_path('src/train_alignment.py', run_name='__main__')
" 2>&1
