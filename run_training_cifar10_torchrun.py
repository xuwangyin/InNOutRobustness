import matplotlib as mpl
mpl.use('Agg')

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from utils.model_normalization import Cifar10Wrapper
import utils.datasets as dl
import utils.models.model_factory_32 as factory
import utils.run_file_helpers as rh
from distutils.util import strtobool

import argparse

def main_training(hps):
    dist.init_process_group(backend='nccl')

    # Get rank and world_size from torchrun environment
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Assert that world_size * batch_size equals 128
    assert world_size * hps.bs == 128, f"world_size ({world_size}) * batch_size ({hps.bs}) must equal 128, got {world_size * hps.bs}"
    
    print(f'Running DDP on rank {rank}.')
    
    # Set device for this process
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    device_ids = None

    # Load model
    model_root_dir = 'Cifar10Models'
    logs_root_dir = 'Cifar10Logs'
    num_classes = 10

    model, model_name, model_config, img_size = factory.build_model(hps.net, num_classes, model_params=hps.model_params)
    model_dir = os.path.join(model_root_dir, model_name)
    log_dir = os.path.join(logs_root_dir, model_name)

    start_epoch, optim_state_dict = rh.load_model_checkpoint(model, model_dir, device, hps)
    model = Cifar10Wrapper(model).to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])

    msda_config = rh.create_msda_config(hps)

    # Load dataset with DDP support
    od_bs = int(hps.od_bs_factor * hps.bs)

    # Set augmentation parameters based on mean choice
    CIFAR10_mean = (0.4913997551666284, 0.48215855929893703, 0.4465309133731618)

    if hps.use_zero_mean:
        custom_augm_parameters = {
            'interpolation': 'bilinear',
            'mean': (0, 0, 0),
            'crop_pct': 0.875
        }
        print(f"Using zero mean augmentation parameters: {custom_augm_parameters}")
    else:
        custom_augm_parameters = {
            'interpolation': 'bilinear',
            'mean': CIFAR10_mean,
            'crop_pct': 0.875
        }
        print(f"Using CIFAR10 mean augmentation parameters: {custom_augm_parameters}")

    id_config = {}
    if hps.dataset == 'cifar10':
        # Create dataset directly for DDP (mimicking dl.get_CIFAR10)
        from torchvision import datasets
        from utils.datasets.augmentations.cifar_augmentation import get_cifar10_augmentation
        from utils.datasets.paths import get_CIFAR10_path
        
        augm_config = {}
        train_transform = get_cifar10_augmentation(type=hps.augm, cutout_window=16, out_size=img_size, 
                                                  config_dict=augm_config, augm_parameters=custom_augm_parameters)
        path = get_CIFAR10_path()
        train_dataset = datasets.CIFAR10(path, train=True, transform=train_transform, download=True)
        
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=hps.bs,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        if id_config is not None:
            id_config['Dataset'] = 'Cifar10'
            id_config['Batch size'] = hps.bs
            id_config['Augmentation'] = augm_config
    elif hps.dataset == 'cifar100':
        print('using cifar100 train loader')
        # Create dataset directly for DDP (mimicking dl.get_CIFAR100)
        from torchvision import datasets
        from utils.datasets.augmentations.cifar_augmentation import get_cifar10_augmentation
        from utils.datasets.paths import get_CIFAR100_path
        
        augm_config = {}
        train_transform = get_cifar10_augmentation(type=hps.augm, cutout_window=8, out_size=img_size, 
                                                  config_dict=augm_config, augm_parameters=custom_augm_parameters)
        path = get_CIFAR100_path()
        train_dataset = datasets.CIFAR100(path, train=True, transform=train_transform, download=True)
        
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=hps.bs,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        if id_config is not None:
            id_config['Dataset'] = 'Cifar100'
            id_config['Batch size'] = hps.bs
            id_config['Augmentation'] = augm_config
    elif hps.dataset == 'semi-cifar10':
        assert False, 'semi-cifar10 not supported in DDP mode'
    else:
        raise ValueError(f'Dataset {hps.dataset} not supported')

    if hps.train_type.lower() in ['ceda', 'acet', 'advacet', 'tradesacet', 'tradesceda']:
        od_config = {}
        loader_config = {'ID config': id_config, 'OD config': od_config}

        if hps.od_dataset == 'tinyImages':
            # Create TinyImages dataset directly for DDP (mimicking dl.get_80MTinyImages)
            from utils.datasets.tinyImages import TinyImagesDataset
            from utils.datasets.augmentations.cifar_augmentation import get_cifar10_augmentation
            
            augm_config = {}
            transform_base = get_cifar10_augmentation(type=hps.augm, cutout_window=16, out_size=img_size, 
                                                     config_dict=augm_config, augm_parameters=custom_augm_parameters)
            tiny_dataset = TinyImagesDataset(transform_base, exclude_cifar=hps.exclude_cifar, 
                                           exclude_cifar10_1=hps.exclude_cifar)
            
            tiny_sampler = DistributedSampler(
                tiny_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
            tiny_train = torch.utils.data.DataLoader(
                tiny_dataset,
                batch_size=od_bs,
                sampler=tiny_sampler,
                num_workers=4,
                pin_memory=True
            )
            
            if od_config is not None:
                od_config['Dataset'] = '80M TinyImages'
                od_config['Batch size'] = od_bs
                od_config['Augmentation'] = augm_config
        elif hps.od_dataset == 'cifar100':
            # Create CIFAR-100 dataset directly for DDP (mimicking dl.get_CIFAR100)
            from torchvision import datasets
            from utils.datasets.augmentations.cifar_augmentation import get_cifar10_augmentation
            from utils.datasets.paths import get_CIFAR100_path
            
            augm_config = {}
            transform = get_cifar10_augmentation(type=hps.augm, cutout_window=8, out_size=img_size, 
                                               config_dict=augm_config, augm_parameters=custom_augm_parameters)
            path = get_CIFAR100_path()
            tiny_dataset = datasets.CIFAR100(path, train=True, transform=transform, download=True)
            
            tiny_sampler = DistributedSampler(
                tiny_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
            tiny_train = torch.utils.data.DataLoader(
                tiny_dataset,
                batch_size=od_bs,
                sampler=tiny_sampler,
                num_workers=4,
                pin_memory=True
            )
            
            if od_config is not None:
                od_config['Dataset'] = 'Cifar100'
                od_config['Batch size'] = od_bs
                od_config['Augmentation'] = augm_config
        elif hps.od_dataset == 'openImages':
            assert False, 'openImages not supported in DDP mode'
    else:
        loader_config = {'ID config': id_config}

    # Create test loader with DDP support
    if hps.dataset == 'cifar10':
        # Create test dataset directly for DDP (mimicking dl.get_CIFAR10)
        from torchvision import datasets
        from utils.datasets.augmentations.cifar_augmentation import get_cifar10_augmentation
        from utils.datasets.paths import get_CIFAR10_path
        
        test_transform = get_cifar10_augmentation(type='none', cutout_window=16, out_size=img_size)
        path = get_CIFAR10_path()
        test_dataset = datasets.CIFAR10(path, train=False, transform=test_transform, download=True)
    elif hps.dataset == 'cifar100':
        print('using cifar100 test loader')
        # Create test dataset directly for DDP (mimicking dl.get_CIFAR100)
        from torchvision import datasets
        from utils.datasets.augmentations.cifar_augmentation import get_cifar10_augmentation
        from utils.datasets.paths import get_CIFAR100_path
        
        test_transform = get_cifar10_augmentation(type='none', cutout_window=8, out_size=img_size)
        path = get_CIFAR100_path()
        test_dataset = datasets.CIFAR100(path, train=False, transform=test_transform, download=True)
    else:
        assert False
    
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=hps.bs,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True
    )

    scheduler_config, optimizer_config = rh.create_optim_scheduler_swa_configs(hps)
    id_attack_config, od_attack_config = rh.create_attack_config(hps, 'cifar10')
    trainer = rh.create_trainer(hps, model, optimizer_config, scheduler_config, device, num_classes,
                                model_dir, log_dir, msda_config=msda_config, model_config=model_config,
                                id_attack_config=id_attack_config, od_attack_config=od_attack_config,
                                use_ddp=True, rank=rank)
    
    # Enable benchmarking for performance
    torch.backends.cudnn.benchmark = True

    # Run training
    if trainer.requires_out_distribution():
        train_loaders, test_loaders = trainer.create_loaders_dict(train_loader, test_loader=test_loader,
                                                                  out_distribution_loader=tiny_train)
        trainer.train(train_loaders, test_loaders, loader_config=loader_config, start_epoch=start_epoch,
                      optim_state_dict=optim_state_dict, device_ids=device_ids)
    else:
        train_loaders, test_loaders = trainer.create_loaders_dict(train_loader, test_loader=test_loader)
        trainer.train(train_loaders, test_loaders, loader_config=loader_config, start_epoch=start_epoch,
                      optim_state_dict=optim_state_dict, device_ids=device_ids)

    dist.destroy_process_group()


def main():
    """Main function for torchrun DDP training."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='CIFAR-10 torchrun DDP Training Script')
    parser.add_argument('--net', type=str, default='ResNet18', help='Resnet18, 34 or 50, WideResNet28')
    parser.add_argument('--model_params', nargs='+', default=[])
    parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10 or semi-cifar10')
    parser.add_argument('--od_dataset', type=str, default='tinyImages',
                        help=('tinyImages or cifar100'))
    parser.add_argument('--exclude_cifar', dest='exclude_cifar', type=lambda x: bool(strtobool(x)),
                        default=True, help='whether to exclude cifar10 from tiny images')
    parser.add_argument('--use_zero_mean', dest='use_zero_mean', type=lambda x: bool(strtobool(x)),
                        default=True, help='If True, use zero mean (0,0,0); if False, use CIFAR10 mean')

    rh.parser_add_commons(parser)
    rh.parser_add_adversarial_commons(parser)
    rh.parser_add_adversarial_norms(parser, 'cifar10')

    hps = parser.parse_args()
    
    # Call main training function
    main_training(hps)


if __name__ == '__main__':
    main()