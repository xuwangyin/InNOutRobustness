
JOB_NAME="cifar100_wideresnet34x10_ratio_training_epoch300_dataparallel_bs128_runpod5090"

export WANDB_NAME=${JOB_NAME}


python -u run_training_cifar10.py \
	--gpu 0  \
	--net wideresnet34x10 \
	--augm autoaugment_cutout \
	--id_steps 10 \
	--od_steps 20 \
	--train_type advacet \
	--epochs 300 \
	--ema True \
	--ema_decay 0.999 \
	--test_epochs 5 \
	--dataset cifar100 \
	--schedule cosine \
	--bs 128 \
	--eps 0.5 \
	--use_zero_mean False \
	--od_eps_factor 2
