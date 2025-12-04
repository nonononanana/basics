"""
OOD Detection Evaluation using Denoising Network
Uses reconstruction error as OOD score
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from meanflow.data.rml_dataset import get_rml_denoising_dataloaders, EXPERIMENT_SETTINGS
from meanflow.models.meanflow_denoising import MeanFlowDenoising
from meanflow.models.unet_denoising import DenoisingUNet

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate OOD Detection with Denoising Network')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained denoising model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to denoising dataset')
    parser.add_argument('--experiment_setting', type=int, default=1, choices=range(1, 13),
                       help='Experiment setting (1-12)')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--method', type=str, default='min_error', 
                       choices=['min_error', 'avg_error', 'improvement', 'snr_improvement'],
                       help='OOD scoring method')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--snr_min', type=int, default=-20,
                       help='Minimum SNR to evaluate')
    parser.add_argument('--snr_max', type=int, default=20,
                       help='Maximum SNR to evaluate')
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> MeanFlowDenoising:
    """Load trained denoising model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract args from checkpoint
    args = checkpoint['args']
    
    # Get number of known classes
    num_classes = len(EXPERIMENT_SETTINGS[args.experiment_setting]['known'])
    
    # Create model with same architecture
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
    
    model = MeanFlowDenoising(
        arch=DenoisingUNet,
        args=args,
        net_configs=net_configs,
        num_classes=num_classes
    )
    
    # Load state dict (use EMA if available)
    if 'ema_state_dict' in checkpoint:
        model.net_ema.load_state_dict(checkpoint['ema_state_dict'])
        logger.info('Loaded EMA model weights')
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Loaded standard model weights')
    
    model.to(device)
    model.eval()
    
    return model


def estimate_snr(signal: torch.Tensor) -> torch.Tensor:
    """
    Estimate SNR of a signal batch using M2M4 estimator.
    Assumes complex signal structure (I/Q channels).
    
    Args:
        signal: [batch, 2, length]
        
    Returns:
        Estimated SNR in dB [batch]
    """
    # Separate I and Q
    i = signal[:, 0, :]
    q = signal[:, 1, :]
    
    # Compute moments
    # M2 = E[|y|^2]
    y_sq = i**2 + q**2
    m2 = torch.mean(y_sq, dim=1)
    
    # M4 = E[|y|^4]
    y_quad = y_sq**2
    m4 = torch.mean(y_quad, dim=1)
    
    # M2M4 Estimate for M-PSK (approximate for others)
    # S = sqrt(2*M2^2 - M4)
    # N = M2 - S
    
    # Calculate signal power
    s_term = 2 * m2**2 - m4
    
    # Handle numerical issues (if 2*M2^2 < M4, estimator fails)
    # This happens for signals with high kurtosis or very low SNR
    # We mask these and assign a low SNR
    valid_mask = s_term > 0
    
    s_est = torch.zeros_like(m2)
    s_est[valid_mask] = torch.sqrt(s_term[valid_mask])
    
    n_est = m2 - s_est
    
    # Compute SNR
    # Clip noise to avoid division by zero
    n_est = torch.clamp(n_est, min=1e-9)
    s_est = torch.clamp(s_est, min=1e-9)
    
    snr_linear = s_est / n_est
    snr_db = 10 * torch.log10(snr_linear)
    
    # Set invalid estimates to a low value
    snr_db[~valid_mask] = -20.0
    
    return snr_db


def compute_ood_score_min_error(
    model: MeanFlowDenoising,
    noisy_signal: torch.Tensor,
    clean_signal: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    OOD score based on minimum reconstruction error across all known classes
    Uses MSE between denoised and clean signal (not noisy input)
    
    Args:
        model: Trained denoising model
        noisy_signal: Noisy signal [batch, 2, 128]
        clean_signal: Clean signal [batch, 2, 128]
        num_classes: Number of known classes
    
    Returns:
        ood_scores: OOD scores [batch] (higher = more likely OOD)
    """
    batch_size = noisy_signal.shape[0]
    device = noisy_signal.device
    
    min_errors = torch.full((batch_size,), float('inf'), device=device)
    
    # Try denoising with each known class
    for class_id in range(num_classes):
        class_labels = torch.full((batch_size,), class_id, dtype=torch.long, device=device)
        
        # Denoise
        with torch.no_grad():
            denoised = model.denoise(
                x_noisy=noisy_signal,
                class_labels=class_labels,
                num_steps=1
            )
        
        # Compute reconstruction error (MSE between denoised and clean signal)
        # For unknown classes, the model cannot denoise well
        errors = F.mse_loss(denoised, clean_signal, reduction='none')
        errors = errors.mean(dim=(1, 2))  # [batch]
        
        # Update minimum errors
        min_errors = torch.minimum(min_errors, errors)
    
    return min_errors


def compute_ood_score_avg_error(
    model: MeanFlowDenoising,
    noisy_signal: torch.Tensor,
    clean_signal: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    OOD score based on average reconstruction error across all known classes
    
    Args:
        model: Trained denoising model
        noisy_signal: Noisy signal [batch, 2, 128]
        clean_signal: Clean signal [batch, 2, 128]
        num_classes: Number of known classes
    
    Returns:
        ood_scores: OOD scores [batch] (higher = more likely OOD)
    """
    batch_size = noisy_signal.shape[0]
    device = noisy_signal.device
    
    denoised_sum = torch.zeros_like(noisy_signal)
    
    # Average denoising over all known classes
    for class_id in range(num_classes):
        class_labels = torch.full((batch_size,), class_id, dtype=torch.long, device=device)
        
        with torch.no_grad():
            denoised = model.denoise(
                x_noisy=noisy_signal,
                class_labels=class_labels,
                num_steps=1
            )
        
        denoised_sum += denoised
    
    denoised_avg = denoised_sum / num_classes
    
    # Compute reconstruction error against clean signal
    errors = F.mse_loss(denoised_avg, clean_signal, reduction='none')
    errors = errors.mean(dim=(1, 2))
    
    return errors


def compute_ood_score_improvement(
    model: MeanFlowDenoising,
    noisy_signal: torch.Tensor,
    clean_signal: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    OOD score based on denoising improvement
    Compares error before and after denoising
    
    Args:
        model: Trained denoising model
        noisy_signal: Noisy signal [batch, 2, 128]
        clean_signal: Clean signal [batch, 2, 128]
        num_classes: Number of known classes
    
    Returns:
        ood_scores: OOD scores [batch] (higher = more likely OOD)
    """
    batch_size = noisy_signal.shape[0]
    device = noisy_signal.device
    
    # Error before denoising
    noise_error = F.mse_loss(noisy_signal, clean_signal, reduction='none')
    noise_error = noise_error.mean(dim=(1, 2))
    
    # Minimum error after denoising
    min_denoised_error = torch.full((batch_size,), float('inf'), device=device)
    
    for class_id in range(num_classes):
        class_labels = torch.full((batch_size,), class_id, dtype=torch.long, device=device)
        
        with torch.no_grad():
            denoised = model.denoise(
                x_noisy=noisy_signal,
                class_labels=class_labels,
                num_steps=1
            )
        
        denoised_error = F.mse_loss(denoised, clean_signal, reduction='none')
        denoised_error = denoised_error.mean(dim=(1, 2))
        
        min_denoised_error = torch.minimum(min_denoised_error, denoised_error)
    
    # Improvement: negative means denoising made it worse (likely OOD)
    improvement = noise_error - min_denoised_error
    
    # OOD score: lower improvement = higher OOD likelihood
    ood_scores = -improvement
    
    return ood_scores


def compute_ood_score_snr_improvement(
    model: MeanFlowDenoising,
    noisy_signal: torch.Tensor,
    clean_signal: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    OOD score based on estimated SNR improvement.
    Uses a blind SNR estimator to calculate SNR before and after denoising.
    Score = -(SNR_after - SNR_before)
    
    Args:
        model: Trained denoising model
        noisy_signal: Noisy signal [batch, 2, 128]
        clean_signal: Clean signal [batch, 2, 128] (Unused)
        num_classes: Number of known classes
    
    Returns:
        ood_scores: OOD scores [batch]
    """
    batch_size = noisy_signal.shape[0]
    device = noisy_signal.device
    
    # Estimate input SNR
    snr_in = estimate_snr(noisy_signal)
    
    max_improvement = torch.full((batch_size,), float('-inf'), device=device)
    
    for class_id in range(num_classes):
        class_labels = torch.full((batch_size,), class_id, dtype=torch.long, device=device)
        
        with torch.no_grad():
            denoised = model.denoise(
                x_noisy=noisy_signal,
                class_labels=class_labels,
                num_steps=1
            )
            
        snr_out = estimate_snr(denoised)
        improvement = snr_out - snr_in
        
        max_improvement = torch.maximum(max_improvement, improvement)
    
    # High improvement = ID (low score)
    # Low improvement = OOD (high score)
    return -max_improvement


def evaluate_ood_detection(
    model: MeanFlowDenoising,
    test_loader: DataLoader,
    method: str,
    device: str
) -> Dict[str, float]:
    """
    Evaluate OOD detection performance
    
    Args:
        model: Trained denoising model
        test_loader: Test data loader
        method: OOD scoring method ('min_error', 'avg_error', 'improvement')
        device: Device to use
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    all_ood_scores = []
    all_labels = []  # 0=known, 1=unknown
    all_snrs = []
    all_modulations = []
    
    # Collect OOD scores
    logger.info(f'Computing OOD scores using method: {method}')
    
    with torch.no_grad():
        for noisy_samples, clean_samples, labels, info in tqdm(test_loader, desc='OOD Evaluation'):
            noisy_samples = noisy_samples.to(device)
            clean_samples = clean_samples.to(device)
            
            # Compute OOD scores
            if method == 'min_error':
                ood_scores = compute_ood_score_min_error(
                    model, noisy_samples, clean_samples, model.num_classes
                )
            elif method == 'avg_error':
                ood_scores = compute_ood_score_avg_error(
                    model, noisy_samples, clean_samples, model.num_classes
                )
            elif method == 'improvement':
                ood_scores = compute_ood_score_improvement(
                    model, noisy_samples, clean_samples, model.num_classes
                )
            elif method == 'snr_improvement':
                ood_scores = compute_ood_score_snr_improvement(
                    model, noisy_samples, clean_samples, model.num_classes
                )
            else:
                raise ValueError(f'Unknown method: {method}')
            
            # Convert labels: known (>=0) -> 0, unknown (-1) -> 1
            is_unknown = (labels == -1).long()
            
            all_ood_scores.append(ood_scores.cpu().numpy())
            all_labels.append(is_unknown.numpy())
            all_snrs.append(info['snr'].numpy())
            all_modulations.extend(info['original_modulation'])
    
    # Concatenate all results
    all_ood_scores = np.concatenate(all_ood_scores)
    all_labels = np.concatenate(all_labels)
    all_snrs = np.concatenate(all_snrs)
    all_modulations = np.array(all_modulations)
    
    # Compute metrics
    auroc = roc_auc_score(all_labels, all_ood_scores)
    aupr = average_precision_score(all_labels, all_ood_scores)
    
    # Compute FPR@95 (False Positive Rate at 95% True Positive Rate)
    fpr, tpr, thresholds = roc_curve(all_labels, all_ood_scores)
    idx_95 = np.argmax(tpr >= 0.95)
    fpr_at_95 = fpr[idx_95] if idx_95 < len(fpr) else 1.0
    
    metrics = {
        'auroc': auroc,
        'aupr': aupr,
        'fpr@95': fpr_at_95
    }
    
    # Per-SNR metrics
    unique_snrs = np.unique(all_snrs)
    for snr in sorted(unique_snrs):
        snr_mask = all_snrs == snr
        if snr_mask.sum() > 0 and all_labels[snr_mask].sum() > 0:  # Has both known and unknown
            try:
                snr_auroc = roc_auc_score(all_labels[snr_mask], all_ood_scores[snr_mask])
                metrics[f'auroc_snr_{int(snr)}'] = snr_auroc
            except:
                pass
    
    # Per-modulation OOD scores (for analysis)
    unique_mods = np.unique(all_modulations)
    for mod in unique_mods:
        mod_mask = all_modulations == mod
        if mod_mask.sum() > 0:
            mod_score = all_ood_scores[mod_mask].mean()
            metrics[f'ood_score_{mod}'] = mod_score
    
    return metrics, all_ood_scores, all_labels, all_snrs, all_modulations


def main():
    args = parse_args()
    
    # Load model
    logger.info(f'Loading model from {args.checkpoint}')
    model = load_model(args.checkpoint, args.device)
    num_classes = model.num_classes
    
    logger.info(f'Model has {num_classes} known classes')
    logger.info(f'Known classes: {EXPERIMENT_SETTINGS[args.experiment_setting]["known"]}')
    logger.info(f'Unknown classes: {EXPERIMENT_SETTINGS[args.experiment_setting]["unknown"]}')
    
    # Load test data
    logger.info('Loading test data...')
    _, _, test_loader = get_rml_denoising_dataloaders(
        data_path=args.data_path,
        experiment_setting=args.experiment_setting,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        snr_range=(args.snr_min, args.snr_max),
        seed=42
    )
    
    # Evaluate OOD detection
    metrics, ood_scores, labels, snrs, modulations = evaluate_ood_detection(
        model=model,
        test_loader=test_loader,
        method=args.method,
        device=args.device
    )
    
    # Print results
    logger.info('\n' + '='*50)
    logger.info('OOD Detection Results')
    logger.info('='*50)
    logger.info(f'Method: {args.method}')
    logger.info(f'AUROC: {metrics["auroc"]:.4f}')
    logger.info(f'AUPR: {metrics["aupr"]:.4f}')
    logger.info(f'FPR@95: {metrics["fpr@95"]:.4f}')
    
    # Print per-SNR results
    logger.info('\nPer-SNR AUROC:')
    snr_keys = sorted([k for k in metrics.keys() if k.startswith('auroc_snr_')])
    for key in snr_keys:
        snr = key.split('_')[-1]
        logger.info(f'  SNR {snr:>3s} dB: {metrics[key]:.4f}')
    
    # Print per-modulation OOD scores
    logger.info('\nMean OOD Scores by Modulation:')
    known_classes = EXPERIMENT_SETTINGS[args.experiment_setting]['known']
    unknown_classes = EXPERIMENT_SETTINGS[args.experiment_setting]['unknown']
    
    logger.info('  Known Classes:')
    for mod in sorted(known_classes):
        if f'ood_score_{mod}' in metrics:
            logger.info(f'    {mod:10s}: {metrics[f"ood_score_{mod}"]:.6f}')
    
    logger.info('  Unknown Classes:')
    for mod in sorted(unknown_classes):
        if f'ood_score_{mod}' in metrics:
            logger.info(f'    {mod:10s}: {metrics[f"ood_score_{mod}"]:.6f}')
    
    logger.info('\n' + '='*50)


if __name__ == '__main__':
    main()
