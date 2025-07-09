#!/usr/bin/env python3

import matplotlib as mpl
mpl.use('Agg')

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
from distutils.util import strtobool
import time
import wandb

from utils.model_normalization import RestrictedImageNetWrapper
import utils.datasets as dl
from utils.datasets.imagenet_subsets import RestrictedImageNet
from utils.datasets.augmentations.imagenet_augmentation import get_imageNet_augmentation
from utils.datasets.paths import get_imagenet_path
import utils.models.model_factory_224 as factory
import utils.run_file_helpers as rh


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def reduce_tensor(tensor, world_size):
    """Reduce tensor across all processes."""
    rt = tensor.clone().float()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def train_worker(rank, world_size, args):
    """Training worker for each GPU."""
    print(f'Running DDP on rank {rank}.')
    setup(rank, world_size)
    
    # Set device for this process
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # Model setup
    model_root_dir = 'RestrictedImageNetModels'
    logs_root_dir = 'RestrictedImageNetLogs'
    num_classes = 9  # RestrictedImageNet has 9 classes
    
    model_params = args.model_params if len(args.model_params) > 0 else None
    if rank == 0:
        print(f'Model parameters: {model_params}')
    
    model, model_name, model_config, img_size = factory.build_model(
        args.net, num_classes, model_params=model_params
    )
    
    model_dir = os.path.join(model_root_dir, model_name)
    log_dir = os.path.join(logs_root_dir, model_name)
    
    # Create directories if they don't exist (only on rank 0)
    if rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
    
    model = RestrictedImageNetWrapper(model).to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    # Dataset setup
    id_config = {}
    
    # Training data loader with DistributedSampler
    augm_config = {}
    train_transform = get_imageNet_augmentation(type=args.augm, out_size=img_size, config_dict=augm_config)
    path = get_imagenet_path()
    
    train_dataset = RestrictedImageNet(
        path=path,
        split='train',
        transform=train_transform,
        balanced=False,
    )
    
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.bs,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Test data loaders - balanced and natural sampling
    test_transform = get_imageNet_augmentation(type='test', out_size=img_size)
    
    test_dataset_balanced = RestrictedImageNet(
        path=path,
        split='val',
        transform=test_transform,
        balanced=True
    )
    
    # Balanced test loader only on rank 0
    if rank == 0:
        from utils.datasets.imagenet_subsets import RestrictedImagenetBalancedSampler
        test_sampler_balanced = RestrictedImagenetBalancedSampler(
            test_dataset_balanced,
            shuffle=False
        )
        
        test_loader_balanced = torch.utils.data.DataLoader(
            test_dataset_balanced,
            batch_size=args.bs,
            sampler=test_sampler_balanced,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        test_loader_balanced = None
    
    test_dataset_natural = RestrictedImageNet(
        path=path,
        split='val',
        transform=test_transform,
        balanced=False
    )
    
    test_sampler_natural = DistributedSampler(
        test_dataset_natural,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    test_loader_natural = torch.utils.data.DataLoader(
        test_dataset_natural,
        batch_size=args.bs,
        sampler=test_sampler_natural,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
                             weight_decay=args.decay, nesterov=args.nesterov)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    
    # Scheduler
    if args.schedule == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Enable benchmarking for performance
    torch.backends.cudnn.benchmark = True
    
    # Flags to log images only once (only on rank 0)
    train_images_logged = False
    test_images_logged = False
    
    def log_images_once(data, target, phase, logged_flag):
        if rank == 0 and not logged_flag:
            images = []
            for i in range(min(8, data.size(0))):  # Log up to 8 images
                img = data[i].cpu()
                label = target[i].cpu().item()
                images.append(wandb.Image(img, caption=f"{phase} Label: {label}"))
            
            wandb.log({f"Sample_{phase}_Images": images}, step=0)
            return True
        return logged_flag
    
    # Initialize Weights & Biases (only on rank 0)
    if rank == 0:
        wandb.init(
            project="InNOutRobustness",
            entity="xuwangyin",
            name=f"restrictedimagenet_resnet50_ddp_training_epoch75_augdefault_naturalsampling_DDP",
            config={
                "model": args.net,
                "epochs": args.epochs,
                "batch_size": args.bs * world_size,  # Total batch size across all GPUs
                "learning_rate": args.lr,
                "weight_decay": args.decay,
                "momentum": args.momentum,
                "optimizer": args.optim,
                "scheduler": args.schedule,
                "augmentation": args.augm,
                "balanced_sampling": False,
                "num_classes": num_classes,
                "num_workers": args.num_workers,
                "world_size": world_size
            }
        )
    
    # Training loop
    if rank == 0:
        print(f'Starting DDP training for {args.epochs} epochs on {world_size} GPUs')
    
    # Initialize moving averages for timing
    data_load_time_ma = 0.0
    model_time_ma = 0.0
    ma_alpha = 0.9  # exponential moving average factor
    
    for epoch in range(args.epochs):
        # Set epoch for sampler to ensure different shuffling each epoch
        train_sampler.set_epoch(epoch)
        
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Time data loading from disk
            data_load_start = time.time()
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            data_load_time = time.time() - data_load_start
            
            # Log training images once (only on rank 0)
            train_images_logged = log_images_once(data, target, "Training", train_images_logged)
            
            # Time model forward pass and backward pass
            model_start = time.time()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            model_time = time.time() - model_start
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update moving averages
            if batch_idx == 0 and epoch == 0:
                # Initialize moving averages with first values
                data_load_time_ma = data_load_time
                model_time_ma = model_time
            else:
                # Exponential moving average update
                data_load_time_ma = ma_alpha * data_load_time_ma + (1 - ma_alpha) * data_load_time
                model_time_ma = ma_alpha * model_time_ma + (1 - ma_alpha) * model_time
            
            if rank == 0 and batch_idx % args.print_freq == 0:
                print(f'Epoch: {epoch+1}/{args.epochs} | Batch: {batch_idx}/{len(train_loader)} | '
                      f'Loss: {loss.item():.6f} | Acc: {100.*correct/total:.2f}% | '
                      f'Data Load: {data_load_time*1000:.2f}ms (MA: {data_load_time_ma*1000:.2f}ms) | '
                      f'Model: {model_time*1000:.2f}ms (MA: {model_time_ma*1000:.2f}ms)')
        
        # Synchronize before evaluation
        dist.barrier()
        
        # Aggregate training metrics across all processes
        train_loss_tensor = torch.tensor(train_loss).cuda()
        correct_tensor = torch.tensor(correct).cuda()
        total_tensor = torch.tensor(total).cuda()
        
        train_loss_tensor = reduce_tensor(train_loss_tensor, world_size)
        correct_tensor = reduce_tensor(correct_tensor, world_size)
        total_tensor = reduce_tensor(total_tensor, world_size)
        
        # Calculate epoch metrics
        epoch_time = time.time() - start_time
        train_acc = correct_tensor.item() / total_tensor.item()
        avg_train_loss = train_loss_tensor.item() / len(train_loader)

        if rank == 0:
            print(f'Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s')
            print(f'Train Loss: {avg_train_loss:.6f} | Train Acc: {train_acc:.2f}%')
        
        # Evaluation phase - balanced sampling (only on rank 0)
        model.eval()
        test_loss_balanced = 0.0
        correct_balanced = 0
        total_balanced = 0
        
        if rank == 0:
            with torch.no_grad():
                for data, target in test_loader_balanced:
                    data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                    
                    # Log test images once (only on rank 0)
                    test_images_logged = log_images_once(data, target, "Test", test_images_logged)
                    
                    output = model(data)
                    test_loss_balanced += criterion(output, target).item()
                    _, predicted = output.max(1)
                    total_balanced += target.size(0)
                    correct_balanced += predicted.eq(target).sum().item()
            
            # Calculate balanced metrics (no aggregation needed since only rank 0)
            test_acc_balanced = correct_balanced / total_balanced
            avg_test_loss_balanced = test_loss_balanced / len(test_loader_balanced)
        else:
            # Other ranks have dummy values
            test_acc_balanced = 0.0
            avg_test_loss_balanced = 0.0
        
        # Evaluation phase - natural sampling
        test_loss_natural = 0.0
        correct_natural = 0
        total_natural = 0
        
        with torch.no_grad():
            for data, target in test_loader_natural:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                
                output = model(data)
                test_loss_natural += criterion(output, target).item()
                _, predicted = output.max(1)
                total_natural += target.size(0)
                correct_natural += predicted.eq(target).sum().item()
        
        # Aggregate test metrics (natural)
        test_loss_natural_tensor = torch.tensor(test_loss_natural).cuda()
        correct_natural_tensor = torch.tensor(correct_natural).cuda()
        total_natural_tensor = torch.tensor(total_natural).cuda()
        
        test_loss_natural_tensor = reduce_tensor(test_loss_natural_tensor, world_size)
        correct_natural_tensor = reduce_tensor(correct_natural_tensor, world_size)
        total_natural_tensor = reduce_tensor(total_natural_tensor, world_size)
        
        test_acc_natural = correct_natural_tensor.item() / total_natural_tensor.item()
        avg_test_loss_natural = test_loss_natural_tensor.item() / len(test_loader_natural)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics to Weights & Biases (only on rank 0)
        if rank == 0:
            wandb.log({
                "Train/Statistics/LR": scheduler.get_last_lr()[0],
                "Train/Statistics/CleanAccuracy": train_acc,
                "Test/Statistics/CleanAccuracy_Balanced": test_acc_balanced,
                "Test/Statistics/CleanAccuracy_Natural": test_acc_natural,
                "Train/Timing/DataLoadTime_MA_ms": data_load_time_ma * 1000,
                "Train/Timing/ModelTime_MA_ms": model_time_ma * 1000,
            }, step=epoch)
        
        # Print epoch results (only on rank 0)
        if rank == 0:
            print(f'Test Loss (Balanced): {avg_test_loss_balanced:.6f} | Test Acc (Balanced): {test_acc_balanced:.2f}%')
            print(f'Test Loss (Natural): {avg_test_loss_natural:.6f} | Test Acc (Natural): {test_acc_natural:.2f}%')
            print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
            print('-' * 80)
    
    # Save final model (only on rank 0)
    if rank == 0:
        final_checkpoint = {
            'epoch': args.epochs,
            'model_state_dict': model.module.state_dict(),  # Use .module to get the original model
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_acc': train_acc,
            'test_acc_balanced': test_acc_balanced,
            'test_acc_natural': test_acc_natural,
            'train_loss': avg_train_loss,
            'test_loss_balanced': avg_test_loss_balanced,
            'test_loss_natural': avg_test_loss_natural
        }
        torch.save(final_checkpoint, os.path.join(model_dir, 'final_model.pth'))
        print(f'Final model saved with test accuracy (balanced): {test_acc_balanced:.2f}%, (natural): {test_acc_natural:.2f}%')
        
        # Finish wandb run
        wandb.finish()
        
        print('Training completed!')
    
    # Clean up
    cleanup()


def main():
    parser = argparse.ArgumentParser(description='RestrictedImageNet DDP Training Script')
    
    # Model parameters
    parser.add_argument('--net', type=str, default='resnet50_timm', 
                       help='Network architecture (default: resnet50_timm)')
    parser.add_argument('--model_params', nargs='+', default=[],
                       help='Additional model parameters')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=75, help='Number of training epochs')
    parser.add_argument('--bs', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--nesterov', type=lambda x: bool(strtobool(x)), default=True, help='Use Nesterov momentum')
    parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adam'], help='Optimizer type')
    parser.add_argument('--schedule', type=str, default='step_lr', choices=['cosine', 'step_lr'], help='Scheduler type')
    parser.add_argument('--augm', type=str, default='default', help='Augmentation type')
    parser.add_argument('--save_freq', type=int, default=10, help='Save model every N epochs')
    parser.add_argument('--print_freq', type=int, default=100, help='Print frequency during training')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of data loading workers per GPU (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Set optimal number of workers if not specified
    if args.num_workers is None:
        args.num_workers = min(os.cpu_count() // torch.cuda.device_count(), 8)  # Divide by number of GPUs
    
    print(f'Using {args.num_workers} data loading workers per GPU')
    
    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError(f"DDP requires at least 2 GPUs, but only {world_size} available")
    
    print(f'Starting DDP training on {world_size} GPUs')
    
    # Spawn processes for each GPU
    mp.spawn(train_worker,
             args=(world_size, args),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    main()
