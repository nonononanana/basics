"""
Visualization script for denoising and reconstruction results.
Plots original noisy signals, clean signals, and denoised+reconstructed signals
with masked regions highlighted.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

from meanflow.data.rml_dataset import get_rml_denoising_dataloaders, EXPERIMENT_SETTINGS
from meanflow.models.meanflow_denoising import MeanFlowDenoising
from meanflow.models.unet_denoising import DenoisingUNet


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize denoising and reconstruction results')
    
    # Model checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='data/RML2016_denoising.pkl',
                       help='Path to denoising dataset')
    parser.add_argument('--experiment_setting', type=int, default=1, choices=range(1, 13),
                       help='Experiment setting (1-12)')
    parser.add_argument('--mask_ratio', type=float, default=0.25,
                       help='Mask ratio used during training (0.0-1.0)')
    
    # Visualization arguments
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='outputs/visualizations',
                       help='Output directory for plots')
    parser.add_argument('--dpi', type=int, default=150,
                       help='DPI for saved figures')
    parser.add_argument('--figsize', type=int, nargs=2, default=[20, 12],
                       help='Figure size (width, height) in inches')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sample selection')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    return args


def load_model(checkpoint_path: str, device: str) -> MeanFlowDenoising:
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded MeanFlowDenoising model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    
    # Get number of classes
    num_classes = len(EXPERIMENT_SETTINGS[args.experiment_setting]['known'])
    
    # Create UNet configuration
    net_configs = {
        'signal_length': 128,
        'in_channels': 2,
        'out_channels': 2,
        'num_classes': num_classes,
        'model_channels': args.model_channels,
        'channel_mult': (1, 2, 2, 2),
        'num_blocks': args.num_blocks,
        'dropout': args.dropout,
        'class_dropout': args.class_dropout,
        'use_attention': True,
        'attention_levels': (2, 3),
        'embedding_type': 'positional'
    }
    
    # Create model
    model = MeanFlowDenoising(
        arch=DenoisingUNet,
        args=args,
        net_configs=net_configs,
        num_classes=num_classes
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from checkpoint: {checkpoint_path}")
    print(f"Model trained for {checkpoint['epoch']} epochs")
    
    return model


def find_mask_region(noisy_signal: np.ndarray, mask_ratio: float) -> Tuple[int, int]:
    """
    Detect the masked region in a noisy signal by finding consecutive zeros.
    
    Args:
        noisy_signal: Noisy signal [2, 128]
        mask_ratio: Expected mask ratio
        
    Returns:
        (start_idx, end_idx) of masked region, or (-1, -1) if no mask detected
    """
    signal_length = noisy_signal.shape[1]
    expected_mask_len = int(signal_length * mask_ratio)
    
    # Check I channel for consecutive zeros
    i_channel = noisy_signal[0, :]
    
    # Find all zero positions
    zero_positions = np.where(np.abs(i_channel) < 1e-9)[0]
    
    if len(zero_positions) == 0:
        return -1, -1
    
    # Find longest consecutive sequence of zeros
    max_start = -1
    max_length = 0
    current_start = zero_positions[0]
    current_length = 1
    
    for i in range(1, len(zero_positions)):
        if zero_positions[i] == zero_positions[i-1] + 1:
            current_length += 1
        else:
            if current_length > max_length:
                max_length = current_length
                max_start = current_start
            current_start = zero_positions[i]
            current_length = 1
    
    # Check last sequence
    if current_length > max_length:
        max_length = current_length
        max_start = current_start
    
    if max_start >= 0:
        return max_start, max_start + max_length
    else:
        return -1, -1


def visualize_sample(
    idx: int,
    noisy_signal: np.ndarray,
    clean_signal: np.ndarray,
    denoised_signal: np.ndarray,
    mask_start: int,
    mask_end: int,
    modulation: str,
    snr: float,
    output_path: str,
    dpi: int = 150
):
    """
    Visualize a single sample with noisy, clean, and denoised signals.
    
    Args:
        idx: Sample index
        noisy_signal: Noisy input signal [2, 128]
        clean_signal: Ground truth clean signal [2, 128]
        denoised_signal: Denoised+reconstructed signal [2, 128]
        mask_start: Start index of masked region
        mask_end: End index of masked region
        modulation: Modulation type
        snr: SNR value
        output_path: Path to save figure
        dpi: DPI for saved figure
    """
    fig, axes = plt.subplots(3, 2, figsize=(20, 12))
    fig.suptitle(f'Sample {idx} - {modulation} at SNR={snr:.0f}dB', fontsize=16, fontweight='bold')
    
    time_steps = np.arange(128)
    
    # Row 1: Noisy Signal (with mask highlighted)
    # I channel
    axes[0, 0].plot(time_steps, noisy_signal[0, :], 'b-', linewidth=1.5, label='I channel')
    if mask_start >= 0 and mask_end >= 0:
        axes[0, 0].axvspan(mask_start, mask_end, alpha=0.3, color='red', label='Masked region')
    axes[0, 0].set_title('Noisy Input - I Channel', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Q channel
    axes[0, 1].plot(time_steps, noisy_signal[1, :], 'r-', linewidth=1.5, label='Q channel')
    if mask_start >= 0 and mask_end >= 0:
        axes[0, 1].axvspan(mask_start, mask_end, alpha=0.3, color='red', label='Masked region')
    axes[0, 1].set_title('Noisy Input - Q Channel', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Row 2: Clean Signal (Ground Truth)
    # I channel
    axes[1, 0].plot(time_steps, clean_signal[0, :], 'g-', linewidth=1.5, label='I channel')
    if mask_start >= 0 and mask_end >= 0:
        axes[1, 0].axvspan(mask_start, mask_end, alpha=0.2, color='yellow', label='Original mask region')
    axes[1, 0].set_title('Clean Ground Truth - I Channel', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Q channel
    axes[1, 1].plot(time_steps, clean_signal[1, :], 'm-', linewidth=1.5, label='Q channel')
    if mask_start >= 0 and mask_end >= 0:
        axes[1, 1].axvspan(mask_start, mask_end, alpha=0.2, color='yellow', label='Original mask region')
    axes[1, 1].set_title('Clean Ground Truth - Q Channel', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Row 3: Denoised+Reconstructed Signal
    # I channel
    axes[2, 0].plot(time_steps, denoised_signal[0, :], 'c-', linewidth=1.5, label='I channel (denoised)')
    axes[2, 0].plot(time_steps, clean_signal[0, :], 'g--', linewidth=1.0, alpha=0.5, label='Ground truth')
    if mask_start >= 0 and mask_end >= 0:
        axes[2, 0].axvspan(mask_start, mask_end, alpha=0.3, color='orange', label='Reconstructed mask region')
    axes[2, 0].set_title('Denoised+Reconstructed - I Channel', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('Time Step')
    axes[2, 0].set_ylabel('Amplitude')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()
    
    # Q channel
    axes[2, 1].plot(time_steps, denoised_signal[1, :], 'y-', linewidth=1.5, label='Q channel (denoised)')
    axes[2, 1].plot(time_steps, clean_signal[1, :], 'm--', linewidth=1.0, alpha=0.5, label='Ground truth')
    if mask_start >= 0 and mask_end >= 0:
        axes[2, 1].axvspan(mask_start, mask_end, alpha=0.3, color='orange', label='Reconstructed mask region')
    axes[2, 1].set_title('Denoised+Reconstructed - Q Channel', fontsize=12, fontweight='bold')
    axes[2, 1].set_xlabel('Time Step')
    axes[2, 1].set_ylabel('Amplitude')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {output_path}")


def create_summary_plot(
    samples_data: List[Dict],
    output_path: str,
    dpi: int = 150
):
    """
    Create a summary plot showing all samples in a grid.
    
    Args:
        samples_data: List of sample dictionaries
        output_path: Path to save summary figure
        dpi: DPI for saved figure
    """
    num_samples = len(samples_data)
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 3 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Denoising and Reconstruction Summary (I Channel)', fontsize=16, fontweight='bold')
    
    time_steps = np.arange(128)
    
    for idx, sample in enumerate(samples_data):
        noisy = sample['noisy']
        clean = sample['clean']
        denoised = sample['denoised']
        mask_start = sample['mask_start']
        mask_end = sample['mask_end']
        modulation = sample['modulation']
        snr = sample['snr']
        
        # Column 1: Noisy
        axes[idx, 0].plot(time_steps, noisy[0, :], 'b-', linewidth=1.0)
        if mask_start >= 0 and mask_end >= 0:
            axes[idx, 0].axvspan(mask_start, mask_end, alpha=0.3, color='red')
        axes[idx, 0].set_ylabel(f'{modulation}\nSNR={snr:.0f}dB', fontsize=9)
        axes[idx, 0].grid(True, alpha=0.3)
        if idx == 0:
            axes[idx, 0].set_title('Noisy Input (masked)', fontsize=11, fontweight='bold')
        if idx == num_samples - 1:
            axes[idx, 0].set_xlabel('Time Step')
        
        # Column 2: Clean
        axes[idx, 1].plot(time_steps, clean[0, :], 'g-', linewidth=1.0)
        if mask_start >= 0 and mask_end >= 0:
            axes[idx, 1].axvspan(mask_start, mask_end, alpha=0.2, color='yellow')
        axes[idx, 1].grid(True, alpha=0.3)
        if idx == 0:
            axes[idx, 1].set_title('Clean Ground Truth', fontsize=11, fontweight='bold')
        if idx == num_samples - 1:
            axes[idx, 1].set_xlabel('Time Step')
        
        # Column 3: Denoised
        axes[idx, 2].plot(time_steps, denoised[0, :], 'c-', linewidth=1.0, label='Denoised')
        axes[idx, 2].plot(time_steps, clean[0, :], 'g--', linewidth=0.8, alpha=0.5, label='GT')
        if mask_start >= 0 and mask_end >= 0:
            axes[idx, 2].axvspan(mask_start, mask_end, alpha=0.3, color='orange')
        axes[idx, 2].grid(True, alpha=0.3)
        if idx == 0:
            axes[idx, 2].set_title('Denoised+Reconstructed', fontsize=11, fontweight='bold')
            axes[idx, 2].legend(loc='upper right', fontsize=8)
        if idx == num_samples - 1:
            axes[idx, 2].set_xlabel('Time Step')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved summary plot to: {output_path}")


def main():
    """Main visualization function"""
    args = parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    # Create data loaders (with masking enabled)
    _, val_loader, _ = get_rml_denoising_dataloaders(
        data_path=args.data_path,
        experiment_setting=args.experiment_setting,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        snr_range=(-20, 20),
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        normalize=False,
        seed=args.seed,
        mask_ratio=args.mask_ratio
    )
    
    print(f"\nCollecting {args.num_samples} samples from validation set...")
    
    # Collect samples
    samples_data = []
    collected = 0
    
    with torch.no_grad():
        for noisy_samples, clean_samples, labels, info in val_loader:
            if collected >= args.num_samples:
                break
            
            # Move to device
            noisy_samples = noisy_samples.to(args.device)
            clean_samples = clean_samples.to(args.device)
            labels = labels.to(args.device)
            
            batch_size = noisy_samples.shape[0]
            
            for i in range(batch_size):
                if collected >= args.num_samples:
                    break
                
                # Get single sample
                noisy = noisy_samples[i:i+1]
                clean = clean_samples[i:i+1]
                label = labels[i:i+1]
                modulation = info['original_modulation'][i]
                snr = info['snr'][i].item()
                
                # Skip OOD samples (label == -1) for cleaner visualization
                if label.item() == -1:
                    continue
                
                # Denoise using the true class label
                denoised = model.denoise(
                    x_noisy=noisy,
                    class_labels=label,
                    num_steps=1
                )
                
                # Convert to numpy
                noisy_np = noisy.cpu().numpy()[0]  # [2, 128]
                clean_np = clean.cpu().numpy()[0]  # [2, 128]
                denoised_np = denoised.cpu().numpy()[0]  # [2, 128]
                
                # Find masked region
                mask_start, mask_end = find_mask_region(noisy_np, args.mask_ratio)
                
                # Store sample data
                samples_data.append({
                    'idx': collected,
                    'noisy': noisy_np,
                    'clean': clean_np,
                    'denoised': denoised_np,
                    'mask_start': mask_start,
                    'mask_end': mask_end,
                    'modulation': modulation,
                    'snr': snr,
                    'label': label.item()
                })
                
                # Visualize individual sample
                output_path = Path(args.output_dir) / f'sample_{collected:02d}_{modulation}_snr{snr:.0f}.png'
                visualize_sample(
                    idx=collected,
                    noisy_signal=noisy_np,
                    clean_signal=clean_np,
                    denoised_signal=denoised_np,
                    mask_start=mask_start,
                    mask_end=mask_end,
                    modulation=modulation,
                    snr=snr,
                    output_path=str(output_path),
                    dpi=args.dpi
                )
                
                collected += 1
                print(f"Processed sample {collected}/{args.num_samples}: {modulation} at SNR={snr:.0f}dB, "
                      f"Mask: [{mask_start}, {mask_end})")
    
    # Create summary plot
    if len(samples_data) > 0:
        summary_path = Path(args.output_dir) / 'summary_all_samples.png'
        create_summary_plot(samples_data, str(summary_path), dpi=args.dpi)
        
        print(f"\n{'='*70}")
        print(f"Visualization complete!")
        print(f"Total samples visualized: {len(samples_data)}")
        print(f"Output directory: {args.output_dir}")
        print(f"{'='*70}")
    else:
        print("No valid samples found for visualization!")


if __name__ == '__main__':
    main()

