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
    # If no masking was applied (mask_ratio == 0), return no mask region
    if mask_ratio == 0.0:
        return -1, -1
    
    signal_length = noisy_signal.shape[1]
    expected_mask_len = int(signal_length * mask_ratio)
    
    # If expected mask length is 0, return no mask
    if expected_mask_len == 0:
        return -1, -1
    
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
    
    # Determine if masking was used
    has_mask = mask_start >= 0 and mask_end >= 0
    task_type = 'Denoising + Reconstruction' if has_mask else 'Denoising Only'
    
    fig.suptitle(f'Sample {idx} - {modulation} at SNR={snr:.0f}dB\n[{task_type}]', 
                 fontsize=16, fontweight='bold')
    
    time_steps = np.arange(128)
    
    # Row 1: Noisy Signal (with mask highlighted if present)
    # I channel
    axes[0, 0].plot(time_steps, noisy_signal[0, :], 'b-', linewidth=1.5, label='I channel')
    if has_mask:
        axes[0, 0].axvspan(mask_start, mask_end, alpha=0.3, color='red', label='Masked region')
    axes[0, 0].set_title('Noisy Input - I Channel', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Q channel
    axes[0, 1].plot(time_steps, noisy_signal[1, :], 'r-', linewidth=1.5, label='Q channel')
    if has_mask:
        axes[0, 1].axvspan(mask_start, mask_end, alpha=0.3, color='red', label='Masked region')
    axes[0, 1].set_title('Noisy Input - Q Channel', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Row 2: Clean Signal (Ground Truth)
    # I channel
    axes[1, 0].plot(time_steps, clean_signal[0, :], 'g-', linewidth=1.5, label='I channel')
    if has_mask:
        axes[1, 0].axvspan(mask_start, mask_end, alpha=0.2, color='yellow', label='Original mask region')
    axes[1, 0].set_title('Clean Ground Truth - I Channel', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Q channel
    axes[1, 1].plot(time_steps, clean_signal[1, :], 'm-', linewidth=1.5, label='Q channel')
    if has_mask:
        axes[1, 1].axvspan(mask_start, mask_end, alpha=0.2, color='yellow', label='Original mask region')
    axes[1, 1].set_title('Clean Ground Truth - Q Channel', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Row 3: Denoised/Reconstructed Signal
    # I channel
    row3_title = 'Denoised+Reconstructed' if has_mask else 'Denoised'
    axes[2, 0].plot(time_steps, denoised_signal[0, :], 'c-', linewidth=1.5, label='I channel (output)')
    axes[2, 0].plot(time_steps, clean_signal[0, :], 'g--', linewidth=1.0, alpha=0.5, label='Ground truth')
    if has_mask:
        axes[2, 0].axvspan(mask_start, mask_end, alpha=0.3, color='orange', label='Reconstructed region')
    axes[2, 0].set_title(f'{row3_title} - I Channel', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('Time Step')
    axes[2, 0].set_ylabel('Amplitude')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()
    
    # Q channel
    axes[2, 1].plot(time_steps, denoised_signal[1, :], 'y-', linewidth=1.5, label='Q channel (output)')
    axes[2, 1].plot(time_steps, clean_signal[1, :], 'm--', linewidth=1.0, alpha=0.5, label='Ground truth')
    if has_mask:
        axes[2, 1].axvspan(mask_start, mask_end, alpha=0.3, color='orange', label='Reconstructed region')
    axes[2, 1].set_title(f'{row3_title} - Q Channel', fontsize=12, fontweight='bold')
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
    
    # Count ID and OOD samples
    num_id = sum(1 for s in samples_data if not s.get('is_ood', False))
    num_ood = sum(1 for s in samples_data if s.get('is_ood', False))
    
    fig.suptitle(f'Denoising and Reconstruction Summary (I Channel)\n'
                 f'ID Samples: {num_id} (black) | OOD Samples: {num_ood} (red)', 
                 fontsize=16, fontweight='bold')
    
    time_steps = np.arange(128)
    
    for idx, sample in enumerate(samples_data):
        noisy = sample['noisy']
        clean = sample['clean']
        denoised = sample['denoised']
        mask_start = sample['mask_start']
        mask_end = sample['mask_end']
        modulation = sample['modulation']
        snr = sample['snr']
        is_ood = sample.get('is_ood', False)
        
        # Choose color based on ID/OOD status
        label_color = 'red' if is_ood else 'black'
        sample_type = 'OOD' if is_ood else 'ID'
        
        # Determine if this sample has masking
        has_mask = mask_start >= 0 and mask_end >= 0
        
        # Column 1: Noisy
        axes[idx, 0].plot(time_steps, noisy[0, :], 'b-', linewidth=1.0)
        if has_mask:
            axes[idx, 0].axvspan(mask_start, mask_end, alpha=0.3, color='red')
        axes[idx, 0].set_ylabel(f'[{sample_type}] {modulation}\nSNR={snr:.0f}dB', 
                                fontsize=9, color=label_color, fontweight='bold' if is_ood else 'normal')
        axes[idx, 0].grid(True, alpha=0.3)
        if idx == 0:
            # Check if any sample has masking for the title
            any_mask = any(s.get('mask_start', -1) >= 0 for s in samples_data)
            col1_title = 'Noisy Input (masked)' if any_mask else 'Noisy Input'
            axes[idx, 0].set_title(col1_title, fontsize=11, fontweight='bold')
        if idx == num_samples - 1:
            axes[idx, 0].set_xlabel('Time Step')
        
        # Column 2: Clean
        axes[idx, 1].plot(time_steps, clean[0, :], 'g-', linewidth=1.0)
        if has_mask:
            axes[idx, 1].axvspan(mask_start, mask_end, alpha=0.2, color='yellow')
        axes[idx, 1].grid(True, alpha=0.3)
        if idx == 0:
            axes[idx, 1].set_title('Clean Ground Truth', fontsize=11, fontweight='bold')
        if idx == num_samples - 1:
            axes[idx, 1].set_xlabel('Time Step')
        
        # Column 3: Denoised
        axes[idx, 2].plot(time_steps, denoised[0, :], 'c-', linewidth=1.0, label='Output')
        axes[idx, 2].plot(time_steps, clean[0, :], 'g--', linewidth=0.8, alpha=0.5, label='GT')
        if has_mask:
            axes[idx, 2].axvspan(mask_start, mask_end, alpha=0.3, color='orange')
        axes[idx, 2].grid(True, alpha=0.3)
        if idx == 0:
            # Check if any sample has masking for the title
            any_mask = any(s.get('mask_start', -1) >= 0 for s in samples_data)
            col3_title = 'Denoised+Reconstructed' if any_mask else 'Denoised'
            axes[idx, 2].set_title(col3_title, fontsize=11, fontweight='bold')
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
    
    print(f"\nCollecting samples from validation set...")
    print("Step 1: Gathering all available samples (ID and OOD)...")
    
    # First pass: collect ID and OOD samples separately
    id_samples = []
    ood_samples = []
    
    with torch.no_grad():
        for noisy_samples, clean_samples, labels, info in val_loader:
            # Move to device
            noisy_samples = noisy_samples.to(args.device)
            clean_samples = clean_samples.to(args.device)
            labels = labels.to(args.device)
            
            batch_size = noisy_samples.shape[0]
            
            for i in range(batch_size):
                sample_dict = {
                    'noisy': noisy_samples[i:i+1],
                    'clean': clean_samples[i:i+1],
                    'label': labels[i:i+1],
                    'modulation': info['original_modulation'][i],
                    'snr': info['snr'][i].item(),
                    'is_ood': labels[i].item() == -1
                }
                
                # Separate ID and OOD samples
                if labels[i].item() == -1:
                    ood_samples.append(sample_dict)
                else:
                    id_samples.append(sample_dict)
    
    print(f"Found {len(id_samples)} ID samples and {len(ood_samples)} OOD samples in validation set")
    
    # Step 2: Randomly select samples from both ID and OOD
    num_id_to_select = min(args.num_samples, len(id_samples))
    num_ood_to_select = min(args.num_samples, len(ood_samples))
    
    print(f"\nStep 2: Randomly selecting samples...")
    print(f"  ID samples to select: {num_id_to_select}")
    print(f"  OOD samples to select: {num_ood_to_select}")
    
    selected_samples = []
    
    # Select ID samples
    if num_id_to_select > 0:
        id_indices = np.random.choice(len(id_samples), size=num_id_to_select, replace=False)
        selected_samples.extend([id_samples[i] for i in id_indices])
    else:
        print("  WARNING: No ID samples available!")
    
    # Select OOD samples
    if num_ood_to_select > 0:
        ood_indices = np.random.choice(len(ood_samples), size=num_ood_to_select, replace=False)
        selected_samples.extend([ood_samples[i] for i in ood_indices])
    else:
        print("  WARNING: No OOD samples available!")
    
    if len(selected_samples) == 0:
        print("ERROR: No samples available for visualization!")
        return
    
    # Print diversity statistics
    print(f"\nSelected sample diversity (Total: {len(selected_samples)}):")
    
    # ID samples statistics
    id_selected = [s for s in selected_samples if not s['is_ood']]
    if len(id_selected) > 0:
        print(f"\n  ID Samples ({len(id_selected)}):")
        id_mod_counts = {}
        id_snr_ranges = {'low': 0, 'mid': 0, 'high': 0}
        for sample in id_selected:
            mod = sample['modulation']
            snr = sample['snr']
            id_mod_counts[mod] = id_mod_counts.get(mod, 0) + 1
            if snr < -5:
                id_snr_ranges['low'] += 1
            elif snr < 10:
                id_snr_ranges['mid'] += 1
            else:
                id_snr_ranges['high'] += 1
        
        print(f"    Modulation types: {len(id_mod_counts)}")
        for mod, count in sorted(id_mod_counts.items()):
            print(f"      {mod}: {count}")
        print(f"    SNR distribution:")
        print(f"      Low SNR (<-5dB): {id_snr_ranges['low']}")
        print(f"      Mid SNR (-5 to 10dB): {id_snr_ranges['mid']}")
        print(f"      High SNR (>10dB): {id_snr_ranges['high']}")
    
    # OOD samples statistics
    ood_selected = [s for s in selected_samples if s['is_ood']]
    if len(ood_selected) > 0:
        print(f"\n  OOD Samples ({len(ood_selected)}):")
        ood_mod_counts = {}
        ood_snr_ranges = {'low': 0, 'mid': 0, 'high': 0}
        for sample in ood_selected:
            mod = sample['modulation']
            snr = sample['snr']
            ood_mod_counts[mod] = ood_mod_counts.get(mod, 0) + 1
            if snr < -5:
                ood_snr_ranges['low'] += 1
            elif snr < 10:
                ood_snr_ranges['mid'] += 1
            else:
                ood_snr_ranges['high'] += 1
        
        print(f"    Modulation types: {len(ood_mod_counts)}")
        for mod, count in sorted(ood_mod_counts.items()):
            print(f"      {mod}: {count}")
        print(f"    SNR distribution:")
        print(f"      Low SNR (<-5dB): {ood_snr_ranges['low']}")
        print(f"      Mid SNR (-5 to 10dB): {ood_snr_ranges['mid']}")
        print(f"      High SNR (>10dB): {ood_snr_ranges['high']}")
    
    # Step 3: Process and visualize selected samples
    total_samples = len(selected_samples)
    print(f"\nStep 3: Processing and visualizing {total_samples} samples...")
    samples_data = []
    
    with torch.no_grad():
        for idx, sample in enumerate(selected_samples):
            noisy = sample['noisy']
            clean = sample['clean']
            label = sample['label']
            modulation = sample['modulation']
            snr = sample['snr']
            is_ood = sample['is_ood']
            
            # For OOD samples, try all known classes and pick best reconstruction
            # For ID samples, use the true class label
            if is_ood:
                # Try all known classes and find the one with minimum reconstruction error
                best_denoised = None
                min_error = float('inf')
                best_class_id = -1
                
                for class_id in range(model.num_classes):
                    hypothesis_label = torch.full_like(label, class_id)
                    denoised_hypothesis = model.denoise(
                        x_noisy=noisy,
                        class_labels=hypothesis_label,
                        num_steps=20
                    )
                    
                    # Compute reconstruction error (blind metric: correlation with noisy input)
                    # Higher correlation = better match to expected clean signal
                    noisy_flat = noisy.view(1, -1)
                    denoised_flat = denoised_hypothesis.view(1, -1)
                    # Normalize
                    noisy_norm = noisy_flat - noisy_flat.mean()
                    denoised_norm = denoised_flat - denoised_flat.mean()
                    correlation = torch.cosine_similarity(noisy_norm, denoised_norm, dim=1).item()
                    
                    # Higher correlation is better, so use negative for minimization
                    error = -correlation
                    
                    if error < min_error:
                        min_error = error
                        best_denoised = denoised_hypothesis
                        best_class_id = class_id
                
                denoised = best_denoised
                ood_label = f"OOD(best:{best_class_id})"
            else:
                # ID sample: use true class label
                denoised = model.denoise(
                    x_noisy=noisy,
                    class_labels=label,
                    num_steps=20
                )
                ood_label = f"ID(class:{label.item()})"
            
            # Convert to numpy
            noisy_np = noisy.cpu().numpy()[0]  # [2, 128]
            clean_np = clean.cpu().numpy()[0]  # [2, 128]
            denoised_np = denoised.cpu().numpy()[0]  # [2, 128]
            
            # Find masked region
            mask_start, mask_end = find_mask_region(noisy_np, args.mask_ratio)
            
            # Store sample data
            samples_data.append({
                'idx': idx,
                'noisy': noisy_np,
                'clean': clean_np,
                'denoised': denoised_np,
                'mask_start': mask_start,
                'mask_end': mask_end,
                'modulation': modulation,
                'snr': snr,
                'label': label.item(),
                'is_ood': is_ood
            })
            
            # Visualize individual sample
            sample_type = "OOD" if is_ood else "ID"
            output_path = Path(args.output_dir) / f'sample_{idx:02d}_{sample_type}_{modulation}_snr{snr:.0f}.png'
            visualize_sample(
                idx=idx,
                noisy_signal=noisy_np,
                clean_signal=clean_np,
                denoised_signal=denoised_np,
                mask_start=mask_start,
                mask_end=mask_end,
                modulation=f"{modulation} [{ood_label}]",
                snr=snr,
                output_path=str(output_path),
                dpi=args.dpi
            )
            
            # Display mask info
            if mask_start >= 0 and mask_end >= 0:
                mask_info = f"Mask:[{mask_start:3d},{mask_end:3d})"
            else:
                mask_info = "No mask      "
            print(f"  [{idx+1}/{total_samples}] {sample_type:3s} | {modulation:8s} | SNR={snr:3.0f}dB | {mask_info}")
    
    # Create summary plot
    if len(samples_data) > 0:
        summary_path = Path(args.output_dir) / 'summary_all_samples.png'
        create_summary_plot(samples_data, str(summary_path), dpi=args.dpi)
        
        # Calculate final statistics
        num_id_final = sum(1 for s in samples_data if not s.get('is_ood', False))
        num_ood_final = sum(1 for s in samples_data if s.get('is_ood', False))
        
        print(f"\n{'='*70}")
        print(f"Visualization complete!")
        print(f"Total samples visualized: {len(samples_data)}")
        print(f"  - ID samples: {num_id_final}")
        print(f"  - OOD samples: {num_ood_final}")
        print(f"Output directory: {args.output_dir}")
        print(f"\nGenerated files:")
        print(f"  - {len(samples_data)} individual sample plots")
        print(f"  - 1 summary plot (summary_all_samples.png)")
        print(f"{'='*70}")
    else:
        print("No valid samples found for visualization!")


if __name__ == '__main__':
    main()

