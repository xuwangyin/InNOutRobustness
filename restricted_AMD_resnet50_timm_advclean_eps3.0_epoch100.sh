JOB_NAME="restrictedimagenet_resnet50_adv_training_epoch100_DDP_noAMP_savefix_ema_withclean_bs32_eps3.0"
export WANDB_NAME=${JOB_NAME}
export HIP_VISIBLE_DEVICES=0,1,2,3
torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node 4 \
  run_training_restrictedimagenet_torchrun.py \
  --net resnet50_timm \
  --augm madry \
  --epochs 100 \
  --ema True \
  --test_epochs 1 \
  --dataset restrictedImagenet \
  --schedule step_lr \
  --bs 32 \
  --od_eps_factor 2 \
  --train_type adversarial \
  --train_clean True \
  --id_steps 7

