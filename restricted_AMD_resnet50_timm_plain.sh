#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -p mi2104x
#SBATCH -J restrictedimagenet_resnet50_plain_training_epoch200
#SBATCH -o slurm_logs/restrictedimagenet_resnet50_plain_training_epoch200_%j.out

JOB_NAME="restrictedimagenet_resnet50_plain_training_epoch200"

export WANDB_NAME="${JOB_NAME}"

python -u run_training_restrictedimagenet.py \
  --gpu 0 1 2 3 \
  --net resnet50_timm \
  --augm madry \
  --epochs 200 \
  --ema False \
  --test_epochs 1 \
  --dataset restrictedImagenet \
  --schedule step_lr \
  --train_type plain 
