#!/usr/bin/env python3

import matplotlib as mpl
mpl.use('Agg')

import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from distutils.util import strtobool
import time
import wandb

from utils.model_normalization import RestrictedImageNetWrapper
import utils.datasets as dl
import utils.models.model_factory_224 as factory
import utils.run_file_helpers as rh


def main():
    parser = argparse.ArgumentParser(description='RestrictedImageNet Training Script')
    
    # Model parameters
    parser.add_argument('--net', type=str, default='resnet50_timm', 
                       help='Network architecture (default: resnet50_timm)')
    parser.add_argument('--model_params', nargs='+', default=[],
                       help='Additional model parameters')
    
    # Dataset parameters
    parser.add_argument('--balanced_sampling', type=lambda x: bool(strtobool(x)), default=False,
                       help='Use balanced sampling for dataset (default: True)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=75, help='Number of training epochs')
    parser.add_argument('--bs', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--nesterov', type=lambda x: bool(strtobool(x)), default=True, help='Use Nesterov momentum')
    parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adam'], help='Optimizer type')
    parser.add_argument('--schedule', type=str, default='step_lr', choices=['cosine', 'step_lr'], help='Scheduler type')
    parser.add_argument('--augm', type=str, default='default', help='Augmentation type')
    parser.add_argument('--gpu', nargs='+', type=int, default=[0], help='GPU IDs to use')
    parser.add_argument('--save_freq', type=int, default=10, help='Save model every N epochs')
    parser.add_argument('--print_freq', type=int, default=100, help='Print frequency during training')
    
    args = parser.parse_args()
    
    # Device setup
    device_ids = None
    if len(args.gpu) == 0:
        device = torch.device('cpu')
        print('Warning! Computing on CPU')
    elif len(args.gpu) == 1:
        device = torch.device(f'cuda:{args.gpu[0]}')
    else:
        device_ids = [int(i) for i in args.gpu]
        device = torch.device(f'cuda:{min(device_ids)}')
    
    # Model setup
    model_root_dir = 'RestrictedImageNetModels'
    logs_root_dir = 'RestrictedImageNetLogs'
    num_classes = 9  # RestrictedImageNet has 9 classes
    
    model_params = args.model_params if len(args.model_params) > 0 else None
    print(f'Model parameters: {model_params}')
    
    model, model_name, model_config, img_size = factory.build_model(
        args.net, num_classes, model_params=model_params
    )
    
    model_dir = os.path.join(model_root_dir, model_name)
    log_dir = os.path.join(logs_root_dir, model_name)
    
    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    model = RestrictedImageNetWrapper(model).to(device)
    
    # Dataset setup
    id_config = {}
    
    # Training data loader
    train_loader = dl.get_restrictedImageNet(
        train=True, 
        batch_size=args.bs, 
        augm_type=args.augm, 
        size=img_size,
        config_dict=id_config, 
        balanced=args.balanced_sampling
    )
    
    # Standard training configuration
    loader_config = {'ID config': id_config}
    
    # Test data loaders - balanced and natural sampling
    test_loader_balanced = dl.get_restrictedImageNet(
        train=False, 
        batch_size=args.bs, 
        augm_type='test', 
        size=img_size, 
        balanced=True
    )
    
    test_loader_natural = dl.get_restrictedImageNet(
        train=False, 
        batch_size=args.bs, 
        augm_type='test', 
        size=img_size, 
        balanced=False
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
    
    # Flags to log images only once
    train_images_logged = False
    test_images_logged = False
    
    def log_images_once(data, target, phase, logged_flag):
        if not logged_flag:
            images = []
            for i in range(min(8, data.size(0))):  # Log up to 8 images
                img = data[i].cpu()
                label = target[i].cpu().item()
                images.append(wandb.Image(img, caption=f"{phase} Label: {label}"))
            
            wandb.log({f"Sample_{phase}_Images": images}, step=0)
            return True
        return logged_flag
    
    # Initialize Weights & Biases
    wandb.init(
        project="InNOutRobustness",
        entity="xuwangyin",
        name=f"restrictedimagenet_resnet50_standard_training_epoch75_augdefault_naturalsampling",
        config={
            "model": args.net,
            "epochs": args.epochs,
            "batch_size": args.bs,
            "learning_rate": args.lr,
            "weight_decay": args.decay,
            "momentum": args.momentum,
            "optimizer": args.optim,
            "scheduler": args.schedule,
            "augmentation": args.augm,
            "balanced_sampling": args.balanced_sampling,
            "num_classes": num_classes
        }
    )
    
    # Training loop
    print(f'Starting training for {args.epochs} epochs')
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Log training images once
            train_images_logged = log_images_once(data, target, "Training", train_images_logged)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % args.print_freq == 0:
                print(f'Epoch: {epoch+1}/{args.epochs} | Batch: {batch_idx}/{len(train_loader)} | '
                      f'Loss: {loss.item():.6f} | Acc: {100.*correct/total:.2f}%')
        
        # Calculate epoch metrics
        epoch_time = time.time() - start_time
        train_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluation phase - balanced sampling
        model.eval()
        test_loss_balanced = 0.0
        correct_balanced = 0
        total_balanced = 0
        
        with torch.no_grad():
            for data, target in test_loader_balanced:
                data, target = data.to(device), target.to(device)
                
                # Log test images once
                test_images_logged = log_images_once(data, target, "Test", test_images_logged)
                
                output = model(data)
                test_loss_balanced += criterion(output, target).item()
                _, predicted = output.max(1)
                total_balanced += target.size(0)
                correct_balanced += predicted.eq(target).sum().item()
        
        test_acc_balanced = correct_balanced / total_balanced
        avg_test_loss_balanced = test_loss_balanced / len(test_loader_balanced)
        
        # Evaluation phase - natural sampling
        test_loss_natural = 0.0
        correct_natural = 0
        total_natural = 0
        
        with torch.no_grad():
            for data, target in test_loader_natural:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                test_loss_natural += criterion(output, target).item()
                _, predicted = output.max(1)
                total_natural += target.size(0)
                correct_natural += predicted.eq(target).sum().item()
        
        test_acc_natural = correct_natural / total_natural
        avg_test_loss_natural = test_loss_natural / len(test_loader_natural)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics to Weights & Biases
        wandb.log({
            "Train/Statistics/LR": scheduler.get_last_lr()[0],
            "Train/Statistics/CleanAccuracy": train_acc,
            "Test/Statistics/CleanAccuracy_Balanced": test_acc_balanced,
            "Test/Statistics/CleanAccuracy_Natural": test_acc_natural,
        }, step=epoch)
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s')
        print(f'Train Loss: {avg_train_loss:.6f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss (Balanced): {avg_test_loss_balanced:.6f} | Test Acc (Balanced): {test_acc_balanced:.2f}%')
        print(f'Test Loss (Natural): {avg_test_loss_natural:.6f} | Test Acc (Natural): {test_acc_natural:.2f}%')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        print('-' * 80)
        
        # # Save model checkpoint
        # if (epoch + 1) % args.save_freq == 0:
        #     checkpoint = {
        #         'epoch': epoch + 1,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #         'train_acc': train_acc,
        #         'test_acc_balanced': test_acc_balanced,
        #         'test_acc_natural': test_acc_natural,
        #         'train_loss': avg_train_loss,
        #         'test_loss_balanced': avg_test_loss_balanced,
        #         'test_loss_natural': avg_test_loss_natural
        #     }
        #     torch.save(checkpoint, os.path.join(model_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        #     print(f'Checkpoint saved at epoch {epoch+1}')
    
    # Save final model
    final_checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
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


if __name__ == '__main__':
    main()
