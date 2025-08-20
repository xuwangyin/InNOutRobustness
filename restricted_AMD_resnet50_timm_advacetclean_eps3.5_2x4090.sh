#!/usr/bin/env bash
###############################################################################
# Usage examples
#   ./submit_resnet.sh           # uses default 0.5
#   ./submit_resnet.sh 0.75      # overrides clean-loss weight
###############################################################################

# 1 ── Parse CLI
CLEAN_WEIGHT="${1:-0.5}"

# 2 ── Derived names/paths
JOB_NAME="restrictedimagenet_resnet50_advacet_training_epoch75_DDP_noAMP_savefix_ema_withclean_bs32_cleanw${CLEAN_WEIGHT}_eps3.5"
LOG_DIR="slurm_logs"
LOG_FILE="${LOG_DIR}/${JOB_NAME}_%j.out"

# ensure log directory exists
mkdir -p "${LOG_DIR}"

export WANDB_NAME=${JOB_NAME}; \
torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node 2 \
  run_training_restrictedimagenet_torchrun.py \
  --net resnet50_timm \
  --augm madry \
  --epochs 75 \
  --ema True \
  --test_epochs 1 \
  --dataset restrictedImagenet \
  --schedule step_lr \
  --bs 64 \
  --eps 3.5 \
  --od_eps_factor 2 \
  --train_type advacet \
  --train_clean True \
  --id_steps 7 \
  --id_steps 10 \
  --clean_weight ${CLEAN_WEIGHT} \

