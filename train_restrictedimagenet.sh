#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -p mi2104x
#SBATCH -J restrictedimagenet_resnet50_standard_training_epoch75_augdefault_naturalsampling
#SBATCH -o slurm_logs/restrictedimagenet_resnet50_standard_training_epoch75_augdefault_naturalsampling.out


python -u train_restrictedimagenet.py --gpu 0 1 2 3
