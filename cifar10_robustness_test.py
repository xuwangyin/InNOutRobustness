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
import utils.adversarial_attacks as aa
from autoattack import AutoAttack

from utils.load_trained_model import load_model
import utils.datasets as dl

model_descriptions = [
    # ('WideResNet34x10', '_temp_Adversarial Training_25-04-2025_02:48:14', 'best_avg', None, False),
    # ('WideResNet34x10', 'Adversarial Training_25-04-2025_21:24:11', '225', None, False),
    ('WideResNet34x10', 'Adversarial Training_25-04-2025_21:24:11/checkpoints/', 'model_bestfid', None, False),
    ('WideResNet34x10', 'Adversarial Training_25-04-2025_21:24:11/checkpoints/', 'model_bestacc', None, False),
    # ('WideResNet34x10', 'cifar10_500k_apgd_asam', 'best_avg', None, False),
    # ('WideResNet34x10', 'cifar10_pgd', 'best_avg', None, False),
    # ('WideResNet34x10', 'cifar10_apgd', 'best_avg', None, False),
    # ('WideResNet34x10', 'cifar10_500k_pgd', 'best_avg', None, False),
    # ('WideResNet34x10', 'cifar10_500k_apgd', 'best_avg', None, False),
]


parser = argparse.ArgumentParser(description='Parse arguments.', prefix_chars='-')

parser.add_argument('--gpu','--list', nargs='+', default=list(range(torch.cuda.device_count())),
                    help='GPU indices, if more than 1 parallel modules will be called')
parser.add_argument('--model_type', type=str, default='WideResNet34x10',
                    help='Model architecture type (e.g., WideResNet34x10)')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Full path to checkpoint (e.g., Adversarial Training_25-04-2025_21:24:11/checkpoints/model_bestfid)')
parser.add_argument('--temperature', type=float, default=None,
                    help='Temperature for softmax (default: None)')
parser.add_argument('--load_temp', action='store_true', default=False,
                    help='Load temperature from checkpoint if available')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'restrictedimagenet', 'imagenet'],
                    help='Dataset to use for testing (cifar10, cifar100, restrictedimagenet, or imagenet)')
parser.add_argument('--distance_type', type=str, default='L2', choices=['L2', 'Linf'],
                    help='Distance type for adversarial attacks (L2 or Linf)')
parser.add_argument('--eps', type=float, default=0.5,
                    help='Epsilon value for adversarial attacks (default: 0.5 for CIFAR10/100, 3.5 for RestrictedImageNet)')

hps = parser.parse_args()

if len(hps.gpu)==0:
    device = torch.device('cpu')
    print('Warning! Computing on CPU')
    num_devices = 1
elif len(hps.gpu)==1:
    device_ids = [int(hps.gpu[0])]
    device = torch.device('cuda:' + str(hps.gpu[0]))
    num_devices = 1
else:
    device_ids = [int(i) for i in hps.gpu]
    device = torch.device('cuda:' + str(min(device_ids)))
    num_devices = len(device_ids)

ROBUSTNESS_DATAPOINTS = 100_000
dataset = hps.dataset

if hps.dataset == 'restrictedimagenet':
    original_parsed_model_type = hps.model_type # Store to see if it was different
    if original_parsed_model_type != 'resnet50_timm':
        print(f"INFO: Dataset is 'restrictedimagenet'. Global model_type is being set to 'resnet50_timm' "
              f"(original parsed value was '{original_parsed_model_type}').")
    hps.model_type = 'resnet50_timm'    

# Set default epsilon based on dataset if not provided
if hps.eps is None:
    if dataset in ['cifar10', 'cifar100']:
        hps.eps = 0.5
    elif dataset == 'restrictedimagenet':
        hps.eps = 3.5
    elif dataset == 'imagenet':
        hps.eps = 3.0
    print(f"Using default epsilon value of {hps.eps} for {dataset}")

bs = 1000 * num_devices

print(f'Testing on {ROBUSTNESS_DATAPOINTS} points from {dataset.upper()} dataset')

# Override model descriptions if command line arguments are provided
if hps.checkpoint is not None:
    # Extract folder and checkpoint name from the full path
    import os
    path_parts = hps.checkpoint.split('/')
    checkpoint_name = path_parts[-1]
    folder = '/'.join(path_parts[:-1])
    if folder:
        folder += '/'  # Add trailing slash if folder is not empty
    
    model_descriptions = [
        (hps.model_type, folder, checkpoint_name, hps.temperature, hps.load_temp),
    ]

for model_idx, (type, folder, checkpoint, temperature, temp) in enumerate(model_descriptions):
    model = load_model(type, folder, checkpoint,
                       temperature, device, load_temp=temp, dataset=dataset)
    model.to(device)

    if len(hps.gpu) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    model.eval()
    print(f'\n\n{folder} {checkpoint}\n ')

    # TODO using augm_type='none' give higher numbers
    restrictedimagenet_augm_type = 'test'

    if dataset == 'cifar10':
        dataloader = dl.get_CIFAR10(False, batch_size=bs, augm_type='none')
    elif dataset == 'cifar100':
        dataloader = dl.get_CIFAR100(False, batch_size=bs, augm_type='none')
    elif dataset == 'restrictedimagenet':
        dataloader = dl.get_restrictedImageNet(train=False, augm_type=restrictedimagenet_augm_type, batch_size=bs, balanced=False)
    elif dataset == 'imagenet':
        dataloader = dl.get_ImageNet(train=False, batch_size=bs, augm_type='test')
    else:
        raise NotImplementedError()

    acc = 0.0
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            out = model(data)
            _, pred = torch.max(out, dim=1)
            acc += torch.sum(pred == target).item() / len(dataloader.dataset)

    print(f'Clean accuracy {acc} (balanced=False)')

    if dataset == 'cifar10':
        dataloader = dl.get_CIFAR10(False, batch_size=ROBUSTNESS_DATAPOINTS, augm_type='none')
    elif dataset == 'cifar100':
        dataloader = dl.get_CIFAR100(False, batch_size=ROBUSTNESS_DATAPOINTS, augm_type='none')
    elif dataset == 'restrictedimagenet':
        dataloader = dl.get_restrictedImageNet(train=False, augm_type=restrictedimagenet_augm_type, batch_size=ROBUSTNESS_DATAPOINTS, balanced=True)
    elif dataset == 'imagenet':
        dataloader = dl.get_ImageNet(train=False, batch_size=ROBUSTNESS_DATAPOINTS, augm_type='test')
    else:
        raise NotImplementedError()
    
    data_iterator = iter(dataloader)
    ref_data, target = next(data_iterator)

    print(f'Distance type: {hps.distance_type}, Eps: {hps.eps}')
    
    # Determine number of classes based on dataset
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'restrictedimagenet':
        num_classes = 9  # RestrictedImageNet has 9 classes
    elif dataset == 'imagenet':
        num_classes = 1000  # ImageNet has 1000 classes
    else:
        raise NotImplementedError(f"Number of classes not defined for dataset: {dataset}")
    
    if hps.distance_type == 'L2':
        attack = AutoAttack(model, device=device, norm='L2', eps=hps.eps, verbose=True)
        # Set the number of target classes for targeted attacks
        attack.apgd_targeted.n_target_classes = num_classes - 1
        attack.fab.n_target_classes = num_classes - 1
        attack.run_standard_evaluation(ref_data, target, bs=bs)
    elif hps.distance_type == 'Linf':
        attack = AutoAttack(model, device=device, norm='Linf', eps=hps.eps, verbose=True)
        # Set the number of target classes for targeted attacks
        attack.apgd_targeted.n_target_classes = num_classes - 1
        attack.fab.n_target_classes = num_classes - 1
        attack.run_standard_evaluation(ref_data, target, bs=bs)

