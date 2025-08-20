#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -p mi2104x
#SBATCH -J restrictedimagenet_resnet50_adv_training_epoch75_DDP_noAMP_savefix_ema_withclean_bs32_4090
#SBATCH -o slurm_logs/restrictedimagenet_resnet50_adv_training_epoch75_DDP_noAMP_savefix_ema_withclean_bs32_4090_%j.out


JOB_NAME="restrictedimagenet_resnet50_adv_training_epoch75_DDP_noAMP_savefix_ema_withclean_bs32_4090"

export WANDB_NAME="${JOB_NAME}"

python -u run_training_restrictedimagenet.py \
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
  --train_type adversarial \
  --train_clean True \
  --id_steps 7
