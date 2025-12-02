"""
Training script for Mean Flow Signal Denoising
Learns to denoise signals conditioned on modulation type
Includes comprehensive denoising evaluation metrics
"""

import os
import sys
import time
import datetime
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Import wandb for experiment tracking
import wandb

# Import custom modules
from meanflow.data.rml_dataset import get_rml_denoising_dataloaders, RML2016DenoisingDataset, EXPERIMENT_SETTINGS
from meanflow.models.meanflow_denoising import MeanFlowDenoising
from meanflow.models.unet_denoising import DenoisingUNet
from meanflow.training import distributed_mode

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Mean Flow for Signal Denoising')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='data/RML2016_denoising.pkl',
                       help='Path to denoising dataset (4-channel format)')
    parser.add_argument('--experiment_setting', type=int, default=1, choices=range(1, 13),
                       help='Experiment setting (1-12) for known/unknown split')
    parser.add_argument('--snr_min', type=int, default=-20,
                       help='Minimum SNR to include')
    parser.add_argument('--snr_max', type=int, default=20,
                       help='Maximum SNR to include')
    
    # Model arguments
    parser.add_argument('--model_channels', type=int, default=64,
                       help='Base channel dimension for UNet')
    parser.add_argument('--num_blocks', type=int, default=2,
                       help='Number of blocks per resolution')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout probability')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--class_dropout', type=float, default=0.1,
                       help='Class dropout for classifier-free guidance')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=512,
                       help='Batch size for evaluation')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='Number of warmup epochs')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                       help='EMA decay rate')
    parser.add_argument('--ema_decays', type=float, nargs='+', default=[0.9999, 0.99999],
                       help='Additional EMA decay rates')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping value')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training')
    
    # Mean flow specific arguments
    parser.add_argument('--norm_eps', type=float, default=1e-3,
                       help='Epsilon for adaptive normalization')
    parser.add_argument('--norm_p', type=float, default=0.75,
                       help='Power for adaptive normalization')
    parser.add_argument('--num_timesteps', type=int, default=1000,
                       help='Number of diffusion timesteps')
    parser.add_argument('--tr_sampler', type=str, default='v1', choices=['v0', 'v1'],
                       help='Time sampler version')
    parser.add_argument('--P_mean_t', type=float, default=0.0,
                       help='Mean for t sampling')
    parser.add_argument('--P_std_t', type=float, default=1.0,
                       help='Std for t sampling')
    parser.add_argument('--P_mean_r', type=float, default=0.0,
                       help='Mean for r sampling')
    parser.add_argument('--P_std_r', type=float, default=1.0,
                       help='Std for r sampling')
    parser.add_argument('--ratio', type=float, default=0.5,
                       help='Ratio for time sampling')
    
    # Data preprocessing arguments
    parser.add_argument('--normalize', action='store_true', default=False,
                       help='Normalize I/Q samples in the dataset')
    
    # Evaluation arguments
    parser.add_argument('--eval_freq', type=int, default=5,
                       help='Evaluation frequency (epochs)')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='outputs/denoising',
                       help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='meanflow-denoising',
                       help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Wandb entity name')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='Wandb run name')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable wandb logging')
    
    args = parser.parse_args()
    return args


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(args) -> MeanFlowDenoising:
    """
    Create the Mean Flow model for denoising
    
    Args:
        args: Command line arguments
    
    Returns:
        MeanFlowDenoising model
    """
    # Get number of known classes for this experiment setting
    num_classes = len(EXPERIMENT_SETTINGS[args.experiment_setting]['known'])
    
    # Create UNet configuration for denoising
    net_configs = {
        'signal_length': 128,
        'in_channels': 2,  # I and Q channels
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
    
    # Create Mean Flow model for denoising
    model = MeanFlowDenoising(
        arch=DenoisingUNet,
        args=args,
        net_configs=net_configs,
        num_classes=num_classes
    )
    
    return model


def compute_denoising_metrics(
    x_clean_pred: torch.Tensor,
    x_clean_true: torch.Tensor,
    x_noisy: torch.Tensor
) -> Dict[str, float]:
    """
    Compute denoising quality metrics
    
    Args:
        x_clean_pred: Predicted clean signal [batch, 2, 128]
        x_clean_true: Ground truth clean signal [batch, 2, 128]
        x_noisy: Input noisy signal [batch, 2, 128]
    
    Returns:
        Dictionary of metrics
    """
    # MSE: Mean Squared Error
    mse = F.mse_loss(x_clean_pred, x_clean_true).item()
    
    # NMSE: Normalized MSE (normalized by signal power)
    signal_power = (x_clean_true ** 2).mean().item()
    nmse = mse / (signal_power + 1e-8)
    nmse_db = 10 * np.log10(nmse + 1e-10)
    
    # SNR Improvement: Compare output SNR to input SNR
    noise_before = x_noisy - x_clean_true
    noise_after = x_clean_pred - x_clean_true
    
    noise_power_before = (noise_before ** 2).mean().item()
    noise_power_after = (noise_after ** 2).mean().item()
    
    snr_before = 10 * np.log10(signal_power / (noise_power_before + 1e-10))
    snr_after = 10 * np.log10(signal_power / (noise_power_after + 1e-10))
    snr_improvement = snr_after - snr_before
    
    # Correlation coefficient between predicted and true clean signals
    # Flatten and compute correlation
    pred_flat = x_clean_pred.view(-1).cpu().numpy()
    true_flat = x_clean_true.view(-1).cpu().numpy()
    
    correlation = np.corrcoef(pred_flat, true_flat)[0, 1]
    
    # PSNR: Peak Signal-to-Noise Ratio
    max_val = max(x_clean_true.abs().max().item(), 1e-8)
    psnr = 10 * np.log10(max_val ** 2 / (mse + 1e-10))
    
    return {
        'mse': mse,
        'nmse': nmse,
        'nmse_db': nmse_db,
        'snr_before': snr_before,
        'snr_after': snr_after,
        'snr_improvement': snr_improvement,
        'correlation': correlation,
        'psnr': psnr
    }


def train_epoch(
    model: MeanFlowDenoising,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    args,
    scaler: Optional[GradScaler] = None
) -> Dict[str, float]:
    """
    Train for one epoch
    
    Args:
        model: Mean Flow denoising model
        train_loader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        args: Command line arguments
        scaler: Gradient scaler for mixed precision
    
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    # Initialize metrics
    total_loss = 0.0
    flow_loss_total = 0.0
    denoising_loss_total = 0.0
    num_batches = 0
    
    # Training loop with progress bar
    progress_bar = tqdm(
        enumerate(train_loader), 
        total=len(train_loader),
        desc=f'Epoch {epoch}',
        leave=True,
        ncols=120
    )
    
    for batch_idx, (noisy_samples, clean_samples, labels, info) in progress_bar:
        # Move to device
        noisy_samples = noisy_samples.to(args.device, non_blocking=True)
        clean_samples = clean_samples.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if args.mixed_precision and scaler is not None:
            with autocast():
                loss_dict = model.forward_with_loss(
                    x_noisy=noisy_samples,
                    x_clean=clean_samples,
                    class_labels=labels,
                    aug_cond=None
                )
                loss = loss_dict['total_loss']
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular forward pass
            loss_dict = model.forward_with_loss(
                x_noisy=noisy_samples,
                x_clean=clean_samples,
                class_labels=labels,
                aug_cond=None
            )
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # Optimizer step
            optimizer.step()
        
        # Update EMA models
        model.update_ema()
        
        # Update metrics
        total_loss += loss.item()
        flow_loss_total += loss_dict['flow_loss'].item()
        denoising_loss_total += loss_dict['denoising_loss'].item()
        num_batches += 1
        
        # Learning rate scheduler step
        if scheduler is not None:
            scheduler.step()
        
        # Update progress bar
        current_loss = total_loss / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Flow': f'{(flow_loss_total / num_batches):.4f}',
            'Denoise': f'{(denoising_loss_total / num_batches):.4f}',
            'lr': f'{current_lr:.2e}'
        })
    
    # Compute average metrics
    metrics = {
        'train/total_loss': total_loss / num_batches,
        'train/flow_loss': flow_loss_total / num_batches,
        'train/denoising_loss': denoising_loss_total / num_batches,
        'train/learning_rate': optimizer.param_groups[0]['lr']
    }
    
    return metrics


def evaluate(
    model: MeanFlowDenoising,
    test_loader: DataLoader,
    epoch: int,
    args
) -> Dict[str, float]:
    """
    Evaluate model on test set with denoising metrics
    
    Args:
        model: Mean Flow denoising model
        test_loader: Test data loader
        epoch: Current epoch
        args: Command line arguments
    
    Returns:
        Metrics dictionary
    """
    model.eval()
    
    # Storage for metrics
    all_mse = []
    all_nmse = []
    all_snr_improvement = []
    all_correlation = []
    all_psnr = []
    
    # Per-SNR metrics
    snr_metrics = {}
    
    # Per-modulation metrics
    modulation_metrics = {}
    
    # Evaluation loop
    eval_progress = tqdm(
        test_loader,
        desc='Evaluation',
        leave=False,
        ncols=80
    )
    
    with torch.no_grad():
        for noisy_samples, clean_samples, labels, info in eval_progress:
            # Move to device
            noisy_samples = noisy_samples.to(args.device)
            clean_samples = clean_samples.to(args.device)
            labels = labels.to(args.device)
            
            # Denoise signals
            clean_pred = model.denoise(
                x_noisy=noisy_samples,
                class_labels=labels,
                num_steps=1
            )
            
            # Compute batch metrics
            batch_metrics = compute_denoising_metrics(
                x_clean_pred=clean_pred,
                x_clean_true=clean_samples,
                x_noisy=noisy_samples
            )
            
            all_mse.append(batch_metrics['mse'])
            all_nmse.append(batch_metrics['nmse'])
            all_snr_improvement.append(batch_metrics['snr_improvement'])
            all_correlation.append(batch_metrics['correlation'])
            all_psnr.append(batch_metrics['psnr'])
            
            # Per-sample metrics for SNR and modulation breakdown
            snr_values = info['snr'].numpy()
            modulations = info['original_modulation']
            
            # Store per-SNR metrics
            for i in range(len(snr_values)):
                snr = int(snr_values[i])
                mod = modulations[i]
                
                # Per-sample metrics
                sample_pred = clean_pred[i:i+1]
                sample_true = clean_samples[i:i+1]
                sample_noisy = noisy_samples[i:i+1]
                
                sample_metrics = compute_denoising_metrics(sample_pred, sample_true, sample_noisy)
                
                # Aggregate by SNR
                if snr not in snr_metrics:
                    snr_metrics[snr] = {'mse': [], 'snr_imp': [], 'corr': []}
                snr_metrics[snr]['mse'].append(sample_metrics['mse'])
                snr_metrics[snr]['snr_imp'].append(sample_metrics['snr_improvement'])
                snr_metrics[snr]['corr'].append(sample_metrics['correlation'])
                
                # Aggregate by modulation
                if mod not in modulation_metrics:
                    modulation_metrics[mod] = {'mse': [], 'snr_imp': [], 'corr': []}
                modulation_metrics[mod]['mse'].append(sample_metrics['mse'])
                modulation_metrics[mod]['snr_imp'].append(sample_metrics['snr_improvement'])
                modulation_metrics[mod]['corr'].append(sample_metrics['correlation'])
    
    # Compile overall metrics
    metrics = {
        'eval/mse': np.mean(all_mse),
        'eval/nmse': np.mean(all_nmse),
        'eval/snr_improvement': np.mean(all_snr_improvement),
        'eval/correlation': np.mean(all_correlation),
        'eval/psnr': np.mean(all_psnr)
    }
    
    # Add per-SNR metrics
    for snr, snr_data in sorted(snr_metrics.items()):
        metrics[f'eval/snr_{snr}/mse'] = np.mean(snr_data['mse'])
        metrics[f'eval/snr_{snr}/snr_improvement'] = np.mean(snr_data['snr_imp'])
        metrics[f'eval/snr_{snr}/correlation'] = np.mean(snr_data['corr'])
    
    # Add per-modulation metrics
    for mod, mod_data in modulation_metrics.items():
        metrics[f'eval/mod_{mod}/mse'] = np.mean(mod_data['mse'])
        metrics[f'eval/mod_{mod}/snr_improvement'] = np.mean(mod_data['snr_imp'])
        metrics[f'eval/mod_{mod}/correlation'] = np.mean(mod_data['corr'])
    
    # Log summary
    logger.info(f'\nDenoising Evaluation Results:')
    logger.info(f'  Overall MSE: {metrics["eval/mse"]:.6f}')
    logger.info(f'  Overall NMSE: {metrics["eval/nmse"]:.6f}')
    logger.info(f'  SNR Improvement: {metrics["eval/snr_improvement"]:.2f} dB')
    logger.info(f'  Correlation: {metrics["eval/correlation"]:.4f}')
    logger.info(f'  PSNR: {metrics["eval/psnr"]:.2f} dB')
    
    # Log per-SNR summary
    logger.info('\nPer-SNR Results:')
    for snr in sorted(snr_metrics.keys()):
        logger.info(f'  SNR {snr:3d} dB: MSE={np.mean(snr_metrics[snr]["mse"]):.6f}, '
                   f'Imp={np.mean(snr_metrics[snr]["snr_imp"]):.2f} dB, '
                   f'Corr={np.mean(snr_metrics[snr]["corr"]):.4f}')
    
    return metrics


def save_checkpoint(
    model: MeanFlowDenoising,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: Dict[str, float],
    args,
    is_best: bool = False
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'args': args
    }
    
    # Save EMA network states
    checkpoint['ema_state_dict'] = model.net_ema.state_dict()
    for i in range(len(model.ema_decays)):
        ema_net = getattr(model, f"net_ema{i + 1}")
        checkpoint[f'ema{i + 1}_state_dict'] = ema_net.state_dict()
    
    # Save EMA update counter
    checkpoint['num_updates'] = model.num_updates.item()
    
    # Save checkpoint
    checkpoint_path = Path(args.output_dir) / f'checkpoint_epoch_{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)
    logger.info(f'Saved checkpoint to {checkpoint_path}')
    
    # Save best model
    if is_best:
        best_path = Path(args.output_dir) / 'best_model.pth'
        torch.save(checkpoint, best_path)
        logger.info(f'Saved best model to {best_path}')


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    set_seed(args.seed)
    
    # Initialize wandb if not disabled
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or f'denoising_exp_{args.experiment_setting}_lr_{args.lr}',
            config=vars(args)
        )
    
    # Log experiment configuration
    logger.info(f'Starting denoising training with configuration:')
    logger.info(f'Experiment setting: {args.experiment_setting}')
    logger.info(f'Known classes: {EXPERIMENT_SETTINGS[args.experiment_setting]["known"]}')
    logger.info(f'Unknown classes: {EXPERIMENT_SETTINGS[args.experiment_setting]["unknown"]}')
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_rml_denoising_dataloaders(
        data_path=args.data_path,
        experiment_setting=args.experiment_setting,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        snr_range=(args.snr_min, args.snr_max),
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        normalize=args.normalize,
        seed=args.seed
    )
    
    # Create model
    model = create_model(args)
    model.to(args.device)
    
    # Log model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model has {num_params:,} trainable parameters')
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler (warmup + cosine annealing)
    warmup_steps = args.warmup_epochs * len(train_loader)
    total_steps = args.epochs * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler() if args.mixed_precision else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_metric = float('inf')  # Lower MSE is better
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load EMA states
        if 'ema_state_dict' in checkpoint:
            model.net_ema.load_state_dict(checkpoint['ema_state_dict'])
        for i in range(len(model.ema_decays)):
            ema_key = f'ema{i + 1}_state_dict'
            if ema_key in checkpoint:
                ema_net = getattr(model, f"net_ema{i + 1}")
                ema_net.load_state_dict(checkpoint[ema_key])
        
        if 'num_updates' in checkpoint:
            model.num_updates.fill_(checkpoint['num_updates'])
        
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f'Resumed from checkpoint at epoch {start_epoch}')
    
    # Training loop
    epoch_progress = tqdm(
        range(start_epoch, args.epochs),
        desc='Training Progress',
        leave=True,
        ncols=100,
        position=0
    )
    
    for epoch in epoch_progress:
        # Train for one epoch
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            args=args,
            scaler=scaler
        )
        
        # Update progress bar
        epoch_progress.set_postfix({
            'Loss': f'{train_metrics["train/total_loss"]:.4f}',
            'LR': f'{train_metrics["train/learning_rate"]:.2e}'
        })
        
        # Log training metrics
        logger.info(f'Training metrics: {train_metrics}')
        if not args.no_wandb:
            wandb.log(train_metrics, step=epoch)
        
        # Evaluate periodically
        if (epoch + 1) % args.eval_freq == 0:
            val_metrics = evaluate(
                model=model,
                test_loader=val_loader,
                epoch=epoch,
                args=args
            )
            
            # Log validation metrics
            logger.info(f'Validation metrics: {val_metrics}')
            if not args.no_wandb:
                val_metrics_wandb = {k.replace('eval/', 'val/'): v for k, v in val_metrics.items()}
                wandb.log(val_metrics_wandb, step=epoch)
            
            # Save checkpoint (lower MSE is better)
            current_metric = val_metrics['eval/mse']
            is_best = current_metric < best_metric
            if is_best:
                best_metric = current_metric
                logger.info(f'New best validation MSE: {best_metric:.6f}')
                epoch_progress.set_description(f'Training Progress (Best MSE: {best_metric:.6f})')
            
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=val_metrics,
                args=args,
                is_best=is_best
            )
    
    # Final evaluation on test set
    logger.info('\nFinal evaluation on test set:')
    final_metrics = evaluate(
        model=model,
        test_loader=test_loader,
        epoch=args.epochs,
        args=args
    )
    
    logger.info(f'Final test set metrics: {final_metrics}')
    if not args.no_wandb:
        test_metrics_wandb = {k.replace('eval/', 'test/'): v for k, v in final_metrics.items()}
        wandb.log(test_metrics_wandb, step=args.epochs)
        wandb.log({f'final/{k}': v for k, v in final_metrics.items()})
        wandb.finish()
    
    logger.info('Training completed!')


if __name__ == '__main__':
    main()

