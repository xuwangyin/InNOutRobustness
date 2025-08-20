#!/usr/bin/env bash
###############################################################################
# CIFAR-10 torchrun DDP training script
# Usage examples
#   ./train_cifar10.sh
###############################################################################

# 1 ── Derived names/paths
JOB_NAME="cifar10_wideresnet34x10_adv_training_epoch300_DDP_bs32"
LOG_DIR="slurm_logs"
LOG_FILE="${LOG_DIR}/${JOB_NAME}_%j.out"

# ensure log directory exists
mkdir -p "${LOG_DIR}"

# 3 ── Submit with sbatch --wrap
sbatch \
  -N 1 \
  -n 1 \
  -t 12:00:00 \
  -p mi2508x \
  -J "${JOB_NAME}" \
  -o "${LOG_FILE}" \
  --wrap="\
    export WANDB_NAME=${JOB_NAME}; \
    torchrun \
      --standalone \
      --nnodes 1 \
      --nproc-per-node 4 \
      run_training_cifar10_torchrun.py \
      --net wideresnet34x10 \
      --augm autoaugment_cutout \
      --id_steps 10 \
      --od_steps 20 \
      --train_type adversarial \
      --epochs 300 \
      --ema True \
      --ema_decay 0.999 \
      --test_epochs 5 \
      --dataset cifar10 \
      --schedule cosine \
      --bs 32 \
      --eps 0.5 \
      --od_eps_factor 2"
