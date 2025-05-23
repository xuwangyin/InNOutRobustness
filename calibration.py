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
import os
from tqdm import tqdm

from utils.load_trained_model import load_model
import utils.datasets as dl
from temperature_wrapper import TemperatureWrapper  # Import TemperatureWrapper

# This script calibrates and evaluates expected calibration error (ECE) for image classification models
# Supports evaluation on CIFAR-10, CIFAR-100, and RestrictedImageNet datasets

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
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'restrictedimagenet'],
                    help='Dataset to use for testing (cifar10, cifar100, or restrictedimagenet)')
parser.add_argument('--output_dir', type=str, default='ece_results',
                    help='Directory to save ECE results and plots')
parser.add_argument('--batch_size', type=int, default=500,
                    help='Batch size for evaluation (will be multiplied by number of GPUs)')
parser.add_argument('--verbose', action='store_true', 
                    help='Print detailed bin-by-bin ECE information to stdout')

hps = parser.parse_args()

# Create output directory
output_dir = hps.output_dir
os.makedirs(output_dir, exist_ok=True)

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

dataset = hps.dataset
bs = hps.batch_size * num_devices


# Set appropriate batch size for RestrictedImageNet if needed
if dataset == 'restrictedimagenet' and hps.batch_size == 500:
    # Reduce batch size for RestrictedImageNet to avoid memory issues
    bs = 128 * num_devices
    print(f"Using smaller batch size ({bs}) for RestrictedImageNet")
    original_parsed_model_type = hps.model_type # Store to see if it was different
    if original_parsed_model_type != 'resnet50_timm':
        print(f"INFO: Dataset is 'restrictedimagenet'. Global model_type is being set to 'resnet50_timm' "
              f"(original parsed value was '{original_parsed_model_type}').")
    hps.model_type = 'resnet50_timm'

print(f'Testing on {dataset.upper()} dataset')

# Function to plot reliability diagram
def plot_reliability_diagram(bin_accuracies, bin_confidences, bin_counts, save_path, ece, dataset, model_type, n_bins=20):
    """Plot reliability diagram similar to the reference image"""
    plt.figure(figsize=(4, 3))  # Half the original size (8, 6)

    # Create bin positions for plotting
    bin_width = 1.0 / n_bins
    bin_positions = np.linspace(0 + bin_width/2, 1 - bin_width/2, n_bins)

    # Filter out empty bins for cleaner plotting
    valid_indices = [i for i, count in enumerate(bin_counts) if count > 0]
    valid_positions = [bin_positions[i] for i in valid_indices]
    valid_accuracies = [bin_accuracies[i] for i in valid_indices]

    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], '--', color='#3498db', linewidth=2, label='Perfect Calibration')

    # Plot calibration bars
    plt.bar(valid_positions, valid_accuracies, width=bin_width*0.9,
            color='#e74c3c', alpha=0.7, edgecolor='#c0392b', linewidth=1.5)

    # Add red box with ECE value
    try:
        # Try to convert to float and format as percentage
        if hasattr(ece, 'item'):
            ece_value = ece.item()
        else:
            ece_value = float(ece)
        ece_text = f'ECE: {ece_value:.2%}'
    except (ValueError, TypeError):
        # If conversion fails, just use as string
        ece_text = f'ECE: {ece}'

    plt.text(0.05, 0.85, ece_text, transform=plt.gca().transAxes,
             bbox=dict(facecolor='#ffcccb', alpha=0.8, boxstyle='round,pad=0.5',
                     edgecolor='#e74c3c'), fontsize=12)

    # Set axis limits and labels
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('confidence', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)

    # Hide the top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches=0)  # Use bbox_inches=0 for tight bounding box
    plt.close()

def print_ece_results(ece, bin_accuracies, bin_confidences, bin_counts, n_bins=20):
    """Print detailed ECE results to stdout"""
    ece_value = ece.item() if hasattr(ece, 'item') else ece
    
    print("\n" + "=" * 80)
    print(f"ECE RESULTS SUMMARY")
    print("=" * 80)
    print(f"Expected Calibration Error (ECE): {ece_value:.6f} ({ece_value*100:.2f}%)")
    
    # Print bin details if verbose
    if hps.verbose:
        print("\nBin-by-bin details:")
        print(f"{'Bin':^15} | {'Count':^8} | {'Confidence':^12} | {'Accuracy':^12} | {'Difference':^12}")
        print("-" * 80)
        
        for i, (count, conf, acc_bin) in enumerate(zip(bin_counts, bin_confidences, bin_accuracies)):
            bin_start = i * (1.0 / n_bins)
            bin_end = (i + 1) * (1.0 / n_bins)
            diff = abs(conf - acc_bin)
            print(f"{bin_start:.2f}-{bin_end:.2f}:^15 | {count:^8d} | {conf:^12.4f} | {acc_bin:^12.4f} | {diff:^12.4f}")
    
    # Print overall calibration statistics
    non_empty_bins = [i for i, count in enumerate(bin_counts) if count > 0]
    if non_empty_bins:
        avg_conf = sum(bin_confidences[i] * bin_counts[i] for i in non_empty_bins) / sum(bin_counts[i] for i in non_empty_bins)
        avg_acc = sum(bin_accuracies[i] * bin_counts[i] for i in non_empty_bins) / sum(bin_counts[i] for i in non_empty_bins)
        max_diff = max(abs(bin_confidences[i] - bin_accuracies[i]) for i in non_empty_bins)
        
        print("\nCalibration Statistics:")
        print(f"Average Confidence: {avg_conf:.4f}")
        print(f"Average Accuracy:   {avg_acc:.4f}")
        print(f"Confidence-Accuracy Gap: {avg_conf-avg_acc:.4f}")
        print(f"Maximum Calibration Error (MCE): {max_diff:.4f}")
    
    print("=" * 80)

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
    # Extract a meaningful name from the checkpoint path
    full_path = folder + checkpoint

    # Check for key identifiers in order of priority
    if 'wandb/run-' in full_path:
        # Extract run ID for wandb runs
        run_id = full_path.split('wandb/run-')[-1].split('/')[0]
        meaningful_name = f"{type}_wandb_{run_id}"
    elif 'AdvACET' in full_path:
        meaningful_name = f"{type}_AdvACET"
    elif 'Adversarial Training' in full_path:
        meaningful_name = f"{type}_AdvTraining"
    else:
        # Fallback to the checkpoint name without extension
        checkpoint_name = checkpoint.replace('.pth', '')
        meaningful_name = f"{type}_{checkpoint_name}"

    # Clean up the name to remove problematic characters
    meaningful_name = meaningful_name.replace(':', '_').replace(' ', '_')
    
    # Include dataset name in the output directory
    model_output_dir = os.path.join(output_dir, f"{dataset}", meaningful_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    print(f'\n\nEvaluating {type} from {folder}{checkpoint}\n')

    # Load the model
    model = load_model(type, folder, checkpoint,
                       temperature, device, load_temp=temp, dataset=dataset)
    model.to(device)

    if len(hps.gpu) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    model.eval()

    # Get the dataloader for clean evaluation
    if dataset == 'cifar10':
        dataloader = dl.get_CIFAR10(False, batch_size=bs, augm_type='none')
    elif dataset == 'cifar100':
        dataloader = dl.get_CIFAR100(False, batch_size=bs, augm_type='none')
    elif dataset == 'restrictedimagenet':
        dataloader = dl.get_restrictedImageNet(train=False, augm_type='test', batch_size=bs, balanced=True)
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")

    # Evaluate clean accuracy
    print("Computing accuracy...")
    acc = 0.0
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Evaluating accuracy"):
            data = data.to(device)
            target = target.to(device)
            out = model(data)
            _, pred = torch.max(out, dim=1)
            acc += torch.sum(pred == target).item() / len(dataloader.dataset)

    print(f'Clean accuracy: {acc:.4f}')

    # Compute ECE on clean data
    print("Computing ECE...")
    ece, bin_accuracies, bin_confidences, bin_counts = TemperatureWrapper.compute_ece(model, dataloader, device)
    
    # Print detailed ECE results to stdout
    print_ece_results(ece, bin_accuracies, bin_confidences, bin_counts)
    
    # Plot reliability diagram for clean data
    print("Generating reliability diagram...")
    # Use simple filename as requested
    save_filename = "reliability_diagram_{}_{}.pdf".format(dataset, type)
    plot_reliability_diagram(
        bin_accuracies, 
        bin_confidences, 
        bin_counts,
        os.path.join(model_output_dir, save_filename),
        ece,
        dataset,
        type,
        n_bins=20
    )
    
    # Save clean results
    print("Saving detailed results...")
    # Use simple filename for results
    results_filename = "ece_results.txt"
    with open(os.path.join(model_output_dir, results_filename), 'w') as f:
        f.write(f"Model: {type}\n")
        f.write(f"Checkpoint: {folder}{checkpoint}\n")
        f.write(f"Dataset: {dataset}\n\n")
        f.write(f"Clean Accuracy: {acc:.4f}\n")
        f.write(f"Expected Calibration Error (ECE): {ece.item():.6f}\n")
        
        # Add bin details
        f.write("\nBin-by-bin details:\n")
        f.write("Bin\tCount\tConfidence\tAccuracy\tDifference\n")
        for i, (count, conf, acc_bin) in enumerate(zip(bin_counts, bin_confidences, bin_accuracies)):
            bin_start = i * (1.0 / 20)  # Assuming 20 bins
            bin_end = (i + 1) * (1.0 / 20)
            diff = abs(conf - acc_bin)
            f.write(f"{bin_start:.2f}-{bin_end:.2f}\t{count}\t{conf:.4f}\t{acc_bin:.4f}\t{diff:.4f}\n")
    
    print(f"Results saved to {model_output_dir}")
