#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -p mi2508x
#SBATCH -o slurm_logs/cifar10_wideresnet34x10_adv_training_epoch300_dataparallel_bs128_AMD.out

JOB_NAME="cifar10_wideresnet34x10_adv_training_epoch300_dataparallel_bs128_AMD"

export WANDB_NAME=${JOB_NAME}
python run_training_cifar10.py \
  --gpu 0 1 2 3 \
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
  --bs 128 \
  --eps 0.5 \
  --od_eps_factor 2
