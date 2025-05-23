import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
import pathlib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse

import utils.datasets as dl
from utils.visual_counterfactual_generation import visual_counterfactuals

parser = argparse.ArgumentParser(description='Parse arguments.', prefix_chars='-')

parser.add_argument('--gpu','--list', nargs='+', default=[0, 1, 2, 3],
                    help='GPU indices, if more than 1 parallel modules will be called')
parser.add_argument('--model_type', type=str, default='WideResNet34x10',
                    help='Model architecture type (e.g., WideResNet34x10)')
parser.add_argument('--checkpoint', nargs='+', default=None,
                    help='Full path to checkpoint(s) (e.g., Adversarial Training_25-04-2025_21:24:11/checkpoints/model_bestfid)')
parser.add_argument('--temperature', type=float, default=None,
                    help='Temperature for softmax (default: None)')
parser.add_argument('--load_temp', action='store_true', default=False,
                    help='Load temperature from checkpoint if available')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'restrictedimagenet'],
                    help='Dataset to use for testing (cifar10, cifar100, or restrictedimagenet)')
parser.add_argument('--distance_type', type=str, default='L2', choices=['L2', 'Linf'],
                    help='Distance type for adversarial attacks (L2 or Linf)')


hps = parser.parse_args()

if hps.dataset == 'restrictedimagenet':
    original_parsed_model_type = hps.model_type # Store to see if it was different
    if original_parsed_model_type != 'resnet50_timm':
        print(f"INFO: Dataset is 'restrictedimagenet'. Global model_type is being set to 'resnet50_timm' "
              f"(original parsed value was '{original_parsed_model_type}').")
    hps.model_type = 'resnet50_timm'

bs = 32
big_model_bs = 10

if len(hps.gpu)==0:
    device = torch.device('cpu')
    print('Warning! Computing on CPU')
elif len(hps.gpu)==1:
    device_ids = None
    device = torch.device('cuda:' + str(hps.gpu[0]))
    bs = bs
    big_model_bs = big_model_bs
else:
    device_ids = [int(i) for i in hps.gpu]
    device = torch.device('cuda:' + str(min(device_ids)))
    bs = bs * len(device_ids)
    big_model_bs = big_model_bs * len(device_ids)

model_descriptions = [
    ('WideResNet34x10', 'cifar10_pgd', 'best_avg', None, False),
    # ('WideResNet34x10', 'cifar10_apgd', 'best_avg', None, False),
    # ('WideResNet34x10', 'cifar10_500k_pgd', 'best_avg', None, False),
    # ('WideResNet34x10', 'cifar10_500k_apgd', 'best_avg', None, False),
    # ('WideResNet34x10', 'cifar10_500k_apgd_asam', 'best_avg', None, False),
]

# Override model descriptions if command line arguments are provided
if hps.checkpoint is not None:
    # Handle list of checkpoints
    import os
    model_descriptions = []
    
    for checkpoint_path in hps.checkpoint:
        # Extract folder and checkpoint name from the full path
        path_parts = checkpoint_path.split('/')
        checkpoint_name = path_parts[-1]
        folder = '/'.join(path_parts[:-1])
        if folder:
            folder += '/'  # Add trailing slash if folder is not empty
        
        model_descriptions.append(
            (hps.model_type, folder, checkpoint_name, hps.temperature, hps.load_temp)
        )

model_batchsize = bs * np.ones(len(model_descriptions), dtype=np.int32)
num_examples = 50
dataset = hps.dataset

# Load the appropriate dataset and labels based on user choice
if dataset == 'cifar10':
    dataloader = dl.get_CIFAR10(False, bs, augm_type='none')
    class_labels = dl.cifar.get_CIFAR10_labels()
    dataset_name = 'cifar10'
elif dataset == 'cifar100':
    dataloader = dl.get_CIFAR100(False, bs, augm_type='none')
    class_labels = dl.cifar.get_CIFAR100_labels()
    dataset_name = 'cifar100'
elif dataset == 'restrictedimagenet':
    restrictedimagenet_augm_type = 'none'
    dataloader = dl.get_restrictedImageNet(train=False, augm_type=restrictedimagenet_augm_type, batch_size=bs, balanced=True)
    class_labels = dl.imagenet_subsets.get_restrictedImageNetLabels()
    dataset_name = 'restrictedimagenet'
    
eval_dir = f'{dataset_name}_counterfactuals'

num_datapoints = len(dataloader.dataset)
norm = 'l2' if hps.distance_type == 'L2' else 'linf'

if norm == 'l1':
    radii = np.linspace(15, 90, 6)
    visual_counterfactuals(model_descriptions, radii, dataloader, model_batchsize, num_examples, class_labels, device,
                           eval_dir, dataset_name, norm='l1', stepsize=5, device_ids=device_ids)
else:
    # Set appropriate radii based on dataset and norm type
    if norm == 'l2':
        if dataset in ['cifar10', 'cifar100']:
            radii = np.linspace(0.5, 3, 6)
        elif dataset == 'restrictedimagenet':
            radii = np.linspace(5, 15, 6)
    else:  # linf
        radii = np.linspace(2/255, 8/255, 6)
    
    visual_counterfactuals(model_descriptions, radii, dataloader, model_batchsize, num_examples, class_labels, device, 
                          eval_dir, dataset_name, norm=norm, device_ids=device_ids)
