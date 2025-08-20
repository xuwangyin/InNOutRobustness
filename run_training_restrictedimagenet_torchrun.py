import matplotlib as mpl
mpl.use('Agg')

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from utils.model_normalization import RestrictedImageNetWrapper
import utils.datasets as dl
from utils.datasets.imagenet_subsets import RestrictedImageNetOD
from utils.datasets.imagenet_subsets import RestrictedImageNet
from utils.datasets.augmentations.imagenet_augmentation import get_imageNet_augmentation
from utils.datasets.paths import get_imagenet_path
import utils.models.model_factory_224 as factory
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
    model_root_dir = f'{hps.task}Models'
    logs_root_dir = f'{hps.task}Logs'
    num_classes = 9  # RestrictedImageNet typically has 9 classes

    if len(hps.model_params) == 0:
        model_params = None
    else:
        model_params = hps.model_params
    
    if rank == 0:
        print(model_params)
    
    model, model_name, model_config, img_size = factory.build_model(hps.net, num_classes, model_params=model_params)
    model_dir = os.path.join(model_root_dir, model_name)
    log_dir = os.path.join(logs_root_dir, model_name)

    start_epoch, optim_state_dict = rh.load_model_checkpoint(model, model_dir, device, hps)
    model = RestrictedImageNetWrapper(model).to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])

    msda_config = rh.create_msda_config(hps)

    # Load dataset with DDP support
    od_bs = int(hps.od_bs_factor * hps.bs)

    id_config = {}
    assert hps.dataset == 'restrictedImagenet', f'Dataset {hps.dataset} not supported'
    
    # Create dataset and sampler manually for DDP
    
    augm_config = {}
    train_transform = get_imageNet_augmentation(type=hps.augm, out_size=img_size, config_dict=augm_config)
    path = get_imagenet_path()
    
    train_dataset = RestrictedImageNet(path=path, split='train', transform=train_transform, balanced=False)
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
        id_config['Dataset'] = 'RestrictedImageNet'
        id_config['Balanced'] = False
        id_config['Batch out_size'] = hps.bs
        id_config['Augmentation'] = augm_config

    if hps.train_type.lower() in ['ceda', 'acet', 'advacet', 'tradesacet', 'tradesceda']:
        # assert False, 'in-distribution training only for now'
        od_config = {}
        loader_config = {'ID config': id_config, 'OD config': od_config}

        if hps.od_dataset == 'restrictedimagenetOD':
            # Create dataset and distributed sampler
            od_dataset = RestrictedImageNetOD(path, split='train', transform=train_transform)
            
            od_sampler = DistributedSampler(
                od_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
            
            tiny_train = torch.utils.data.DataLoader(
                od_dataset, 
                batch_size=hps.bs,
                sampler=od_sampler, 
                num_workers=8
            )
            
            if od_config is not None:
                od_config['Dataset'] = 'RestrictedImageNetOD'
                od_config['Batch out_size'] = od_bs
                od_config['Augmentation'] = augm_config
        elif hps.od_dataset == 'imagenet':
            assert False, 'imagenet not supported'
            tiny_train = dl.get_ImageNet(train=True, batch_size=od_bs, augm_type=hps.augm, size=img_size,
                                        exclude_restricted=True, config_dict=od_config)
        elif hps.od_dataset == 'openImages':
            assert False, 'openImages not supported'
            tiny_train = dl.get_openImages('train', batch_size=od_bs, shuffle=True, augm_type=hps.augm, 
                                         size=img_size, exclude_dataset='restrictedimagenet', config_dict=od_config)
        else:
            raise ValueError(f'OD dataset {hps.od_dataset} not supported')
    else:
        loader_config = {'ID config': id_config}

    # Create test loader with DDP support
    test_transform = get_imageNet_augmentation(type='test', out_size=img_size)
    test_dataset = RestrictedImageNet(path=path, split='val', transform=test_transform, balanced=False)
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
    id_attack_config, od_attack_config = rh.create_attack_config(hps, 'restrictedImagenet')
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
    parser = argparse.ArgumentParser(description='RestrictedImageNet torchrun DDP Training Script')
    parser.add_argument('--net', type=str, default='resnet50_timm', help='ResNet18, 34, 50 or 101')
    parser.add_argument('--model_params', nargs='+', default=[])
    parser.add_argument('--dataset', type=str, default='restrictedimagenet', help='restrictedimagenet')
    parser.add_argument('--od_dataset', type=str, default='restrictedimagenetOD',
                        help=('restrictedimagenetOD, imagenet or openImages'))
    parser.add_argument('--task', type=str, default='RestrictedImageNet',
                        help='Task name used for model and log directory naming')
    
    rh.parser_add_commons(parser)
    rh.parser_add_adversarial_commons(parser)
    rh.parser_add_adversarial_norms(parser, 'restrictedimagenet')
    
    hps = parser.parse_args()
    
    # Call main training function
    main_training(hps)


if __name__ == '__main__':
    main()
