#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -p mi2104x
#SBATCH -J restrictedimagenet_resnet50_plain_training_epoch75_DDP
#SBATCH -o slurm_logs/restrictedimagenet_resnet50_plain_training_epoch75_DDP_%j.out

JOB_NAME="restrictedimagenet_resnet50_plain_training_epoch75_DDP"

export WANDB_NAME="${JOB_NAME}"

python -u run_training_restrictedimagenet.py \
  --net resnet50_timm \
  --augm madry \
  --epochs 75 \
  --ema False \
  --test_epochs 1 \
  --dataset restrictedImagenet \
  --schedule step_lr \
  --train_type plain \
  --bs 32
