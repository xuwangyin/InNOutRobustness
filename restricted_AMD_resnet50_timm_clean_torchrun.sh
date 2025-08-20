#!/usr/bin/env bash
###############################################################################
# Usage examples
#   ./submit_resnet.sh           # uses default 0.5
#   ./submit_resnet.sh 0.75      # overrides clean-loss weight
###############################################################################

# 1 ── Parse CLI
CLEAN_WEIGHT="${1:-0.5}"

# 2 ── Derived names/paths
JOB_NAME="restrictedimagenet_resnet50_plain_training_epoch75_DDP_noAMP_ema"
export WANDB_NAME=${JOB_NAME}
torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node 4 \
  run_training_restrictedimagenet_torchrun.py \
  --net resnet50_timm \
  --augm madry \
  --epochs 75 \
  --ema True \
  --test_epochs 1 \
  --dataset restrictedImagenet \
  --schedule step_lr \
  --bs 32 \
  --eps 3.5 \
  --od_eps_factor 2 \
  --train_type plain \
  --train_clean True \
  --clean_weight ${CLEAN_WEIGHT} \
  --id_steps 7

