python -u run_training_restrictedimagenet.py \
  --gpu 0 \
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
  --train_type adversarial \
  --train_clean True \
  --id_steps 7 \
  2>&1 | tee training_restricted_h100_robustnessresnet_clean.log
