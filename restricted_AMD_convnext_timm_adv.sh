#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -p mi2104x
#SBATCH -J restrictedimagenet_convnexttiny_adv_training_epoch75_DDP_noAMP
#SBATCH -o slurm_logs/restrictedimagenet_convnexttiny_adv_training_epoch75_DDP_noAMP_%j.out

JOB_NAME="restrictedimagenet_convnexttiny_adv_training_epoch75_DDP_noAMP"

export WANDB_NAME="${JOB_NAME}"

python -u run_training_restrictedimagenet.py \
  --net convnext_tiny \
  --augm madry \
  --epochs 75 \
  --ema True \
  --test_epochs 1 \
  --dataset restrictedImagenet \
  --schedule step_lr \
  --bs 32 \
  --eps 3.5 \
  --od_eps_factor 2 \
  --train_type adversarial \
  --train_clean False \
  --id_steps 7 \
  --lr 0.01 \
  --model_params pretrained True use_grn False
