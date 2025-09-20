"""
Training script for Mean Flow Modulation Classification
Implements open-set recognition with energy-based OOD detection
Includes wandb tracking and comprehensive evaluation metrics
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

# Import sklearn metrics for evaluation
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, f1_score
from sklearn.metrics import classification_report

# Import custom modules
from data.rml_dataset import get_rml_dataloaders, RML2016Dataset, EXPERIMENT_SETTINGS
from models.meanflow_modulation import MeanFlowModulation
from models.unet_modulation import ModulationUNet
from training import distributed_mode

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Mean Flow for Modulation Classification')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='data/RML2016.10a_dict.pkl',
                       help='Path to RML2016.10a dataset')
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
    parser.add_argument('--use_arcface', action='store_true', default=True,
                       help='Use ArcFace loss for better class separation')
    parser.add_argument('--arcface_margin', type=float, default=0.5,
                       help='ArcFace margin parameter')
    parser.add_argument('--arcface_scale', type=float, default=15.0,
                       help='ArcFace scale parameter')
    
    # Energy loss hyperparameters
    parser.add_argument('--lambda_rec', type=float, default=1.0,
                       help='Weight for reconstruction loss')
    parser.add_argument('--lambda_arc', type=float, default=0.5,
                       help='Weight for ArcFace loss')
    parser.add_argument('--lambda_pos', type=float, default=0.5,
                       help='Weight for positive energy loss')
    parser.add_argument('--lambda_neg', type=float, default=0.5,
                       help='Weight for negative energy loss')
    parser.add_argument('--lambda_rank', type=float, default=0.1,
                       help='Weight for soft ranking loss')
    parser.add_argument('--lambda_cls', type=float, default=0.1,
                       help='Weight for classification loss via per-class energies and learnable thresholds')
    parser.add_argument('--lambda_anchor', type=float, default=0.1,
                       help='Weight for per-class threshold anchor loss (quantile-target)')
    parser.add_argument('--margin_pos', type=float, default=-2.2,
                       help='Margin for positive energy (should be more negative, e.g., -2.2)')
    parser.add_argument('--margin_neg', type=float, default=-1.5,
                       help='Margin for negative energy (should be less negative, e.g., -1.5)')
    parser.add_argument('--rank_margin', type=float, default=0.5,
                       help='Margin delta for soft ranking loss')
    parser.add_argument('--rank_beta', type=float, default=10.0,
                       help='Beta parameter for soft ranking loss')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=2048,
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
    
    # Evaluation arguments
    parser.add_argument('--eval_freq', type=int, default=5,
                       help='Evaluation frequency (epochs)')
    parser.add_argument('--energy_temperature', type=float, default=1.0,
                       help='Temperature for energy scoring')
    parser.add_argument('--energy_threshold', type=float, default=None,
                       help='Energy threshold for OOD detection (auto-tuned if None)')
    parser.add_argument('--target_fpr', type=float, default=0.05,
                       help='Target false positive rate for threshold tuning on validation')
    parser.add_argument('--use_learned_thresholds', action='store_true', default=True,
                       help='Use learned per-class thresholds for classification and OOD rejection during evaluation')
    parser.add_argument('--use_learned_margins', action='store_true', default=True,
                       help='Use constrained learned margins for energy losses (m_pos,m_neg)')
    parser.add_argument('--auto_lambda', action='store_true', default=True,
                       help='Enable uncertainty-based auto-weighting for losses')
    parser.add_argument('--per_class_anchor', action='store_true', default=True,
                       help='Enable per-class threshold anchor to energy quantiles')
    parser.add_argument('--eval_with_synthetic_negatives', action='store_true', default=True,
                       help='On validation/test, compute OOD metrics using synthetic negatives from the loader if available')
    parser.add_argument('--anchor_quantile', type=float, default=0.2,
                       help='Quantile of per-class energy to anchor thresholds to')
    parser.add_argument('--anchor_delta', type=float, default=0.2,
                       help='Subtract delta from quantile for target threshold')
    parser.add_argument('--anchor_momentum', type=float, default=0.9,
                       help='EMA momentum for per-class quantile tracking')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='outputs/modulation',
                       help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='meanflow-modulation',
                       help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Wandb entity name')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='Wandb run name')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable wandb logging')
    
    # Distributed training
    parser.add_argument('--distributed', action='store_true',
                       help='Use distributed training')
    parser.add_argument('--dist_url', type=str, default='env://',
                       help='URL for distributed training')
    parser.add_argument('--world_size', type=int, default=1,
                       help='Number of distributed processes')
    parser.add_argument('--rank', type=int, default=0,
                       help='Rank of the process')
    
    # Additional augmentation arguments
    parser.add_argument('--use_edm_aug', action='store_true',
                       help='Use EDM augmentation')
    
    args = parser.parse_args()
    return args


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(args) -> MeanFlowModulation:
    """
    Create the Mean Flow model for modulation classification
    
    Args:
        args: Command line arguments
    
    Returns:
        MeanFlowModulation model
    """
    # Get number of known classes for this experiment setting
    num_classes = len(EXPERIMENT_SETTINGS[args.experiment_setting]['known'])
    
    # Create UNet configuration
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
        'embedding_type': 'positional',
        'class_embed_dim': 128
    }
    
    # Create Mean Flow model
    model = MeanFlowModulation(
        arch=ModulationUNet,
        args=args,
        net_configs=net_configs,
        num_classes=num_classes,
        use_arcface=args.use_arcface,
        arcface_margin=args.arcface_margin,
        arcface_scale=args.arcface_scale,
        energy_temperature=args.energy_temperature
    )
    
    return model


def train_epoch(
    model: MeanFlowModulation,
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
        model: Mean Flow model
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
    reconstruction_loss = 0.0
    arcface_loss = 0.0
    pos_energy_loss = 0.0
    neg_energy_loss = 0.0
    rank_loss = 0.0
    classification_loss = 0.0
    anchor_loss = 0.0
    num_batches = 0
    
    # Training loop with progress bar
    progress_bar = tqdm(
        enumerate(train_loader), 
        total=len(train_loader),
        desc=f'Epoch {epoch}',
        leave=True,
        ncols=120
    )
    
    for batch_idx, (pos_samples, neg_samples, labels, info) in progress_bar:
        # Move to device
        pos_samples = pos_samples.to(args.device)
        if neg_samples is not None:
            neg_samples = neg_samples.to(args.device)
        labels = labels.to(args.device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if args.mixed_precision and scaler is not None:
            with autocast():
                loss_dict = model.forward_with_loss(
                    x_pos=pos_samples,
                    x_neg=neg_samples,
                    class_labels=labels,
                    aug_cond=None,
                    lambda_rec=args.lambda_rec,
                    lambda_arc=args.lambda_arc,
                    lambda_pos=args.lambda_pos,
                    lambda_neg=args.lambda_neg,
                    lambda_rank=args.lambda_rank,
                    lambda_cls=args.lambda_cls,
                    lambda_anchor=args.lambda_anchor,
                    margin_pos=args.margin_pos,
                    margin_neg=args.margin_neg,
                    rank_margin=args.rank_margin,
                    rank_beta=args.rank_beta
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
                x_pos=pos_samples,
                x_neg=neg_samples,
                class_labels=labels,
                aug_cond=None,
                lambda_rec=args.lambda_rec,
                lambda_arc=args.lambda_arc,
                lambda_pos=args.lambda_pos,
                lambda_neg=args.lambda_neg,
                lambda_rank=args.lambda_rank,
                lambda_cls=args.lambda_cls,
                lambda_anchor=args.lambda_anchor,
                margin_pos=args.margin_pos,
                margin_neg=args.margin_neg,
                rank_margin=args.rank_margin,
                rank_beta=args.rank_beta
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
        model.update_ema()  #  看看
        
        # Update metrics
        total_loss += loss.item()
        reconstruction_loss += loss_dict['reconstruction_loss'].item()
        if args.use_arcface:
            arcface_loss += loss_dict['arcface_loss'].item()
        pos_energy_loss += loss_dict['pos_energy_loss'].item()
        neg_energy_loss += loss_dict['neg_energy_loss'].item()
        if 'rank_loss' in loss_dict:
            rank_loss += loss_dict['rank_loss'].item()
        if 'classification_loss' in loss_dict:
            classification_loss += loss_dict['classification_loss'].item()
        if 'anchor_loss' in loss_dict:
            anchor_loss += loss_dict['anchor_loss'].item()
        num_batches += 1
        
        # Learning rate scheduler step
        if scheduler is not None:
            scheduler.step()
        
        # Update progress bar with current metrics
        current_loss = total_loss / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Rec': f'{(reconstruction_loss / num_batches):.4f}',
            'Pos': f'{(pos_energy_loss / num_batches):.4f}',
            'Neg': f'{(neg_energy_loss / num_batches):.4f}',
            'lr': f'{current_lr:.2e}',
            'Arc': f'{(arcface_loss / num_batches):.4f}' if args.use_arcface else '0.0000',

        })
    
    # Compute average metrics
    metrics = {
        'train/total_loss': total_loss / num_batches,
        'train/reconstruction_loss': reconstruction_loss / num_batches,
        'train/arcface_loss': arcface_loss / num_batches if args.use_arcface else 0.0,
        'train/pos_energy_loss': pos_energy_loss / num_batches,
        'train/neg_energy_loss': neg_energy_loss / num_batches,
        'train/rank_loss': rank_loss / num_batches,
        'train/learning_rate': optimizer.param_groups[0]['lr']
    }
    if args.lambda_cls > 0:
        metrics['train/classification_loss'] = classification_loss / num_batches
    if args.lambda_anchor > 0:
        metrics['train/anchor_loss'] = anchor_loss / num_batches
    
    return metrics


def evaluate(
    model: MeanFlowModulation,
    test_loader: DataLoader,
    epoch: int,
    args,
    energy_threshold: Optional[float] = None
) -> Tuple[Dict[str, float], float]:
    """
    Evaluate model on test set with open-set metrics
    
    Args:
        model: Mean Flow model
        test_loader: Test data loader
        epoch: Current epoch
        args: Command line arguments
        energy_threshold: Energy threshold for OOD detection
    
    Returns:
        Metrics dictionary and optimal energy threshold
    """
    model.eval()
    
    # Storage for predictions and labels
    all_predictions = []
    all_labels = []
    all_energies = []
    all_is_unknown = []
    all_original_modulations = []  # Track original modulation for per-class metrics
    all_best_scores = []  # used when args.use_learned_thresholds
    
    # Evaluation loop with progress bar
    eval_progress = tqdm(
        test_loader,
        desc=f'Evaluation',
        leave=False,
        ncols=80
    )
    
    with torch.no_grad():
        for pos_samples, neg_samples, labels, info in eval_progress:
            # Move to device (only use positive samples for evaluation)
            signals = pos_samples.to(args.device)
            labels = labels.to(args.device)
            is_unknown = info['is_unknown']
            
            # Compute class energies and predictions without rejection
            class_energies = model.compute_energy_score(signals, return_per_class=True, use_ema=True)
            predictions = class_energies.argmin(dim=1)

            # Compute OOD energy-like scores
            if args.use_learned_thresholds:
                # Use learned thresholds to form scores but don't use their rejection yet
                _, best_scores, _ = model.classify_with_learned_threshold(signals, use_ema=True)
                # Higher values should mean more OOD for AUROC/AUPR, so negate best_scores
                energy_scores = -best_scores
                all_best_scores.append(best_scores.cpu())
            else:
                energy_scores = model.compute_energy_score(signals, use_ema=True)
            
            # Store results
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())  # from dataset
            all_energies.append(energy_scores.cpu())
            all_is_unknown.append(is_unknown)  # from dataset
            all_original_modulations.extend(info.get('original_modulation', []))

            # Optionally evaluate synthetic negatives as OOD for val/test
            if args.eval_with_synthetic_negatives and neg_samples is not None:
                neg_signals = neg_samples.to(args.device)
                if args.use_learned_thresholds:
                    _, neg_best_scores, _ = model.classify_with_learned_threshold(neg_signals, use_ema=True)
                    neg_scores = -neg_best_scores  # higher = more OOD
                else:
                    neg_scores = model.compute_energy_score(neg_signals, use_ema=True)
                # Append as unknowns
                all_predictions.append(torch.full((neg_signals.shape[0],), -1, dtype=torch.long))
                all_labels.append(torch.full((neg_signals.shape[0],), -1, dtype=torch.long))
                all_energies.append(neg_scores.cpu())
                all_is_unknown.append(torch.ones(neg_signals.shape[0], dtype=torch.bool))
            
            # Update progress bar with current stats
            eval_progress.set_postfix({
                'Batch': f'{len(all_predictions)}'
            })
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_energies = torch.cat(all_energies).numpy()
    all_is_unknown = torch.cat(all_is_unknown).numpy()
    all_original_modulations = np.array(all_original_modulations) if all_original_modulations else None
    if args.use_learned_thresholds and all_best_scores:
        all_best_scores = torch.cat(all_best_scores).numpy()
    else:
        all_best_scores = None
    
    # Separate known and unknown samples
    known_mask = ~all_is_unknown
    unknown_mask = all_is_unknown
    
    # Compute closed-set accuracy (only on known samples)
    known_predictions = all_predictions[known_mask]
    known_labels = all_labels[known_mask]
    closed_set_accuracy = (known_predictions == known_labels).mean()  # 注： 与OOD阈值无关，假装数据集只有known classes
    
    # Compute AUROC for OOD detection (only if we have unknown samples)
    # Create binary labels: 0 for known, 1 for unknown
    ood_labels = all_is_unknown.astype(int)
    
    # Check if we have both known and unknown samples (for AUROC computation)
    has_unknown = unknown_mask.sum() > 0
    has_known = known_mask.sum() > 0
    
    if has_unknown and has_known:
        # Use scores where higher = more likely OOD for AUROC/AUPR
        auc_scores = all_energies
        auroc = roc_auc_score(ood_labels, auc_scores)
        precision, recall, _ = precision_recall_curve(ood_labels, auc_scores)
        aupr = auc(recall, precision)
    else:
        # No unknown samples (e.g., validation set) - AUROC/AUPR undefined
        auroc = np.nan
        aupr = np.nan
        logger.info('No unknown samples in dataset - AUROC/AUPR not computed')
    
    # Find optimal energy threshold if not provided
    if energy_threshold is None and not args.use_learned_thresholds:
        if has_unknown:
            # Test set: find threshold that maximizes F1 score
            thresholds = np.percentile(all_energies, np.linspace(0, 100, 100))
            best_f1 = 0
            best_threshold = thresholds[0]
            
            for threshold in thresholds:
                # Apply threshold - energies are negative, so reject when GREATER than threshold
                # (greater means less negative, closer to 0, which indicates OOD)
                predicted_unknown = all_energies > threshold
                
                # Compute F1 score
                f1 = f1_score(ood_labels, predicted_unknown)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            energy_threshold = best_threshold
            logger.info(f'Optimal energy threshold: {energy_threshold:.4f} (F1: {best_f1:.4f})')
        else:
            # Validation set: use target FPR on known samples only
            if known_mask.sum() > 0:
                # Set threshold to achieve target FPR (e.g., 5%)
                target_fpr = getattr(args, 'target_fpr', 0.05)
                # Find the (100 - target_fpr*100) percentile of known energies
                percentile = (1.0 - target_fpr) * 100
                energy_threshold = np.percentile(all_energies[known_mask], percentile)
                logger.info(f'Energy threshold set for {target_fpr:.1%} FPR: {energy_threshold:.4f}')
            else:
                # Fallback to median
                energy_threshold = np.median(all_energies)
                logger.info(f'Using median energy as threshold: {energy_threshold:.4f}')
    
    # Apply rejection for final predictions
    final_predictions = all_predictions.copy()
    if args.use_learned_thresholds and all_best_scores is not None:
        # Reject when best score < 0 (below per-class threshold)
        rejected = all_best_scores < 0
        final_predictions[rejected] = -1
        # For logging consistency, set energy_threshold to 0.0 under learned-threshold regime
        energy_threshold = 0.0 if energy_threshold is None else energy_threshold
    else:
        # Reject when energy is GREATER than threshold (less negative, closer to 0)
        rejected = all_energies > energy_threshold
        final_predictions[rejected] = -1
    
    # Compute open-set accuracy
    # Correct if: (known and correctly classified) or (unknown and rejected)
    correct_known = known_mask & (final_predictions == all_labels)
    correct_unknown = unknown_mask & (final_predictions == -1)
    open_set_accuracy = (correct_known | correct_unknown).mean()
    
    # Compute per-class accuracy for known classes (using class index)
    class_accuracies = {}
    for class_idx in range(model.num_classes): # Model expects 0-8 for 9 known classes
        class_mask = (all_labels == class_idx) & known_mask # all_labels are already mapped to 0-8 for known classes
        if class_mask.sum() > 0:
            class_acc = (final_predictions[class_mask] == class_idx).mean()
            class_accuracies[f'class_{class_idx}'] = class_acc
    
    # Compute per-modulation accuracy (using original modulation names)
    modulation_accuracies = {}
    unknown_class_metrics = {}
    
    if all_original_modulations is not None:
        # EXPERIMENT_SETTINGS already imported at top of file
        unique_modulations = np.unique(all_original_modulations)
        
        known_classes = EXPERIMENT_SETTINGS[args.experiment_setting]['known']
        unknown_classes = EXPERIMENT_SETTINGS[args.experiment_setting]['unknown']
        
        for mod in unique_modulations:
            mod_mask = all_original_modulations == mod
            if mod_mask.sum() > 0:
                # For known classes: check if correctly classified
                # For unknown classes: check if correctly rejected (energy > threshold)
                if mod in known_classes:
                    # Known class - check classification accuracy
                    mod_label = known_classes.index(mod)
                    mod_acc = (final_predictions[mod_mask] == mod_label).mean()
                else:
                    # Unknown class - check if correctly rejected
                    mod_acc = (final_predictions[mod_mask] == -1).mean()
                
                modulation_accuracies[f'{mod}'] = mod_acc
        
        # Compute per-unknown-class OOD detection metrics
        for unknown_cls in unknown_classes:
            unknown_mask_cls = all_original_modulations == unknown_cls
            if unknown_mask_cls.sum() > 0:
                # Get energies for this specific unknown class
                unknown_energies = all_energies[unknown_mask_cls]
                
                # Combine with known energies for AUROC
                combined_energies = np.concatenate([all_energies[known_mask], unknown_energies])
                combined_labels = np.concatenate([
                    np.zeros(known_mask.sum()),  # Known = 0
                    np.ones(len(unknown_energies))  # Unknown = 1
                ])
                
                # Calculate metrics for this unknown class
                # Use scores where higher = more OOD directly
                auroc_cls = roc_auc_score(combined_labels, combined_energies)
                precision_cls, recall_cls, _ = precision_recall_curve(combined_labels, combined_energies)
                aupr_cls = auc(recall_cls, precision_cls)
                
                # Detection accuracy at current threshold
                if args.use_learned_thresholds and all_best_scores is not None:
                    unknown_best_scores = all_best_scores[unknown_mask_cls]
                    detected = unknown_best_scores < 0
                else:
                    detected = unknown_energies > energy_threshold
                detection_rate = detected.mean()
                
                unknown_class_metrics[f'ood_{unknown_cls}_auroc'] = auroc_cls
                unknown_class_metrics[f'ood_{unknown_cls}_aupr'] = aupr_cls
                unknown_class_metrics[f'ood_{unknown_cls}_detection_rate'] = detection_rate
                unknown_class_metrics[f'ood_{unknown_cls}_mean_energy'] = unknown_energies.mean()
                unknown_class_metrics[f'ood_{unknown_cls}_samples'] = unknown_mask_cls.sum()
    
    # Compute confusion matrix for known classes
    known_final_predictions = final_predictions[known_mask]
    known_cm = confusion_matrix(
        known_labels,
        known_final_predictions,
        labels=list(range(model.num_classes))
    )
    
    # Detection metrics
    true_positive_rate = (unknown_mask & rejected).sum() / unknown_mask.sum() if unknown_mask.sum() > 0 else 0
    false_positive_rate = (known_mask & rejected).sum() / known_mask.sum() if known_mask.sum() > 0 else 0
    
    # Calculate TKR (True Known Rate)
    # TKR = TK / K = (Known samples not rejected) / Total known samples
    tkr = (known_mask & ~rejected).sum() / known_mask.sum() if known_mask.sum() > 0 else 0
    
    # Calculate F1-score for OOD detection (only if we have unknown samples)
    if has_unknown:
        ood_f1 = f1_score(ood_labels, rejected)
    else:
        ood_f1 = 0.0  # No unknowns, F1 undefined

    # Compile metrics
    metrics = {
        'eval/OA (Open-Set Accuracy)': open_set_accuracy,
        'eval/closed_set_accuracy': closed_set_accuracy,
        'eval/auroc': auroc,
        'eval/aupr': aupr,
        'eval/F1_score_OOD': ood_f1,
        'eval/TKR': tkr,
        'eval/TUR (TPR)': true_positive_rate,
        'eval/FPR': false_positive_rate,
        'eval/energy_threshold': energy_threshold,
        'eval/mean_energy_known': all_energies[known_mask].mean() if known_mask.sum() > 0 else 0,
        'eval/mean_energy_unknown': all_energies[unknown_mask].mean() if unknown_mask.sum() > 0 else 0,
    }
    
    # Add per-class accuracies (by class index)
    for class_name, acc in class_accuracies.items():
        metrics[f'eval/accuracy_{class_name}'] = acc
    
    # Add per-modulation accuracies (by modulation name)
    if all_original_modulations is not None:
        for mod_name, acc in modulation_accuracies.items():
            metrics[f'eval/modulation_{mod_name}_accuracy'] = acc
        
        # Add per-unknown-class OOD metrics
        for metric_name, value in unknown_class_metrics.items():
            metrics[f'eval/{metric_name}'] = value
    
    # Log confusion matrix
    logger.info(f'Confusion Matrix (Known Classes):\n{known_cm}')
    
    # Log per-modulation accuracies
    if all_original_modulations is not None:
        # EXPERIMENT_SETTINGS already imported at top of file
        
        logger.info('\nPer-Modulation Accuracies:')
        known_classes = EXPERIMENT_SETTINGS[args.experiment_setting]['known']
        unknown_classes = EXPERIMENT_SETTINGS[args.experiment_setting]['unknown']
        
        logger.info('  Known Classes:')
        for mod in known_classes:
            if f'{mod}' in modulation_accuracies:
                logger.info(f'    {mod}: {modulation_accuracies[f"{mod}"]:.4f}')
        
        logger.info('  Unknown Classes (OOD Detection):')
        for mod in unknown_classes:
            if f'{mod}' in modulation_accuracies:
                logger.info(f'    {mod}: {modulation_accuracies[f"{mod}"]:.4f}')
                if f'ood_{mod}_auroc' in unknown_class_metrics:
                    logger.info(f'      - AUROC: {unknown_class_metrics[f"ood_{mod}_auroc"]:.4f}')
                    logger.info(f'      - Detection Rate: {unknown_class_metrics[f"ood_{mod}_detection_rate"]:.4f}')
    
    return metrics, energy_threshold


def save_checkpoint(
    model: MeanFlowModulation,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: Dict[str, float],
    energy_threshold: float,
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
        'energy_threshold': energy_threshold,
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
            name=args.wandb_name or f'exp_{args.experiment_setting}_lr_{args.lr}',
            config=vars(args)
        )
    
    # Log experiment configuration
    logger.info(f'Starting training with configuration:')
    logger.info(f'Experiment setting: {args.experiment_setting}')
    logger.info(f'Known classes: {EXPERIMENT_SETTINGS[args.experiment_setting]["known"]}')
    logger.info(f'Unknown classes: {EXPERIMENT_SETTINGS[args.experiment_setting]["unknown"]}')
    
    # Create data loaders with train/val/test split
    train_loader, val_loader, test_loader = get_rml_dataloaders(
        data_path=args.data_path,
        experiment_setting=args.experiment_setting,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        snr_range=(args.snr_min, args.snr_max),
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        normalize=True,
        seed=args.seed,
        precompute_negatives=True
    )
    
    # Create model
    model = create_model(args)
    model.to(args.device)
    
    # Log model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model has {num_params:,} trainable parameters')
    
    # Create optimizer with weight decay for regularization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),  # Adam parameters are hardcoded
        weight_decay=args.weight_decay  # L2 regularization
    )
    
    # Create learning rate scheduler
    # Warmup + cosine annealing
    warmup_steps = args.warmup_epochs * len(train_loader)
    total_steps = args.epochs * len(train_loader)
    
    def lr_lambda(step):
        # return from 0 to 1 with cosine annealing
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine annealing
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler() if args.mixed_precision else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_metric = 0.0  # Will track best closed_set_accuracy for validation
    energy_threshold = args.energy_threshold
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load EMA network states if available
        if 'ema_state_dict' in checkpoint:
            model.net_ema.load_state_dict(checkpoint['ema_state_dict'])
            logger.info('Loaded main EMA network weights')
        
        # Load additional EMA networks
        for i in range(len(model.ema_decays)):
            ema_key = f'ema{i + 1}_state_dict'
            if ema_key in checkpoint:
                ema_net = getattr(model, f"net_ema{i + 1}")
                ema_net.load_state_dict(checkpoint[ema_key])
                logger.info(f'Loaded EMA{i + 1} network weights')
        
        # Load EMA update counter if available
        if 'num_updates' in checkpoint:
            model.num_updates.fill_(checkpoint['num_updates'])
            logger.info(f'Resumed EMA update counter at {checkpoint["num_updates"]}')
        
        start_epoch = checkpoint['epoch'] + 1
        energy_threshold = checkpoint.get('energy_threshold', energy_threshold)
        logger.info(f'Resumed from checkpoint at epoch {start_epoch}')
    
    # Training loop with overall progress bar
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
        
        # Update overall progress bar with key metrics
        epoch_progress.set_postfix({
            'Loss': f'{train_metrics["train/total_loss"]:.4f}',
            'LR': f'{train_metrics["train/learning_rate"]:.2e}'
        })
        
        # Log training metrics
        logger.info(f'Training metrics: {train_metrics}')
        if not args.no_wandb:
            wandb.log(train_metrics, step=epoch)
        
        # Evaluate on validation set periodically
        if (epoch + 1) % args.eval_freq == 0:
            val_metrics, energy_threshold = evaluate( # 看看没有label的情况下怎么做推理
                model=model,
                test_loader=val_loader,  # Use validation set for model selection
                epoch=epoch,
                args=args,
                energy_threshold=energy_threshold
            )
            
            # Log validation metrics
            logger.info(f'Validation metrics: {val_metrics}')
            if not args.no_wandb:
                # Log with val/ prefix to distinguish from test metrics
                val_metrics_wandb = {k.replace('eval/', 'val/'): v for k, v in val_metrics.items()}
                wandb.log(val_metrics_wandb, step=epoch)
            
            # Save checkpoint based on validation performance (use closed_set_accuracy since no unknowns in val)
            is_best = val_metrics['eval/closed_set_accuracy'] > best_metric
            if is_best:
                best_metric = val_metrics['eval/closed_set_accuracy']
                logger.info(f'New best validation closed-set accuracy: {best_metric:.4f}')
                # Update progress bar to show best performance
                epoch_progress.set_description(f'Training Progress (Best Acc: {best_metric:.4f})')
            
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=val_metrics,
                energy_threshold=energy_threshold,
                args=args,
                is_best=is_best
            )
    
    # Final evaluation on test set (held-out, unseen during training)
    logger.info('\nFinal evaluation on test set:')
    final_metrics, _ = evaluate(
        model=model,
        test_loader=test_loader,
        epoch=args.epochs,
        args=args,
        energy_threshold=energy_threshold
    )
    
    logger.info(f'Final test set metrics: {final_metrics}')
    if not args.no_wandb:
        # Log test metrics with test/ prefix
        test_metrics_wandb = {k.replace('eval/', 'test/'): v for k, v in final_metrics.items()}
        wandb.log(test_metrics_wandb, step=args.epochs)
        wandb.log({f'final/{k}': v for k, v in final_metrics.items()})
        wandb.finish()
    
    logger.info('Training completed!')


if __name__ == '__main__':
    main()
