#!/bin/bash

mkdir -p slurm_logs

# Define configuration strings (each quoted)
CONFIGS=(
  # "--train_type adversarial --train_clean True"
  "--train_type adversarial --train_clean False"
  "--train_type advacet --train_clean True"
  # "--train_type advacet --train_clean False"
)

# Loop over configs and submit each as a separate SLURM job
for i in "${!CONFIGS[@]}"; do
  CONFIG="${CONFIGS[$i]}"

  # Extract values for naming
  TRAIN_CLEAN=$(echo "$CONFIG" | grep -oP '(?<=--train_clean )\w+')
  TRAIN_TYPE=$(echo "$CONFIG" | grep -oP '(?<=--train_type )\w+')

  # Compose job name and logfile
  JOB_NAME="restricted_${TRAIN_TYPE}_clean${TRAIN_CLEAN}"
  LOGFILE="slurm_logs/${JOB_NAME}.log"

  echo "Submitting $JOB_NAME with config: $CONFIG"

  sbatch <<EOF
#!/bin/bash
#SBATCH -J $JOB_NAME
#SBATCH -o $LOGFILE
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -p mi2104x

python run_training_restrictedimagenet.py \
  --gpu 0 1 2 3 \
  --net ResNet50 \
  --augm madry \
  --stepsize 0.7 \
  --epochs 75 \
  --ema False \
  --ema_decay 0.999 \
  --test_epochs 5 \
  --dataset restrictedImagenet \
  --schedule step_lr \
  --eps 3.5 \
  --od_eps_factor 2 \
  $CONFIG
EOF

done
