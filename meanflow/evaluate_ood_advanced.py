"""
Advanced OOD Detection Evaluation for Denoising Networks

Implements two noise-robust OOD detection methods:
1. Signal Domain Residual Spectrum Analysis
2. Feature Domain Distance (Bottleneck Consistency)
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from scipy import signal as scipy_signal

from meanflow.data.rml_dataset import get_rml_denoising_dataloaders, EXPERIMENT_SETTINGS
from meanflow.models.meanflow_denoising import MeanFlowDenoising
from meanflow.models.unet_denoising import DenoisingUNet

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedOODDetector:
    """
    Advanced OOD Detection using:
    1. Signal Domain Residual Spectrum Analysis
    2. Feature Domain Distance (Bottleneck Consistency)
    """
    
    def __init__(self, model: MeanFlowDenoising):
        """
        Initialize the advanced OOD detector
        
        Args:
            model: Trained denoising model
        """
        self.model = model
        self.bottleneck_features = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to extract bottleneck features"""
        def hook_fn(name):
            def hook(module, input, output):
                self.bottleneck_features[name] = output.detach()
            return hook
        
        # Register hook on the first middle block (bottleneck)
        if hasattr(self.model.net_ema, 'middle') and len(self.model.net_ema.middle) > 0:
            self.model.net_ema.middle[0].register_forward_hook(hook_fn('bottleneck'))
            logger.info("Registered hook on net_ema.middle[0]")
        elif hasattr(self.model.net, 'middle') and len(self.model.net.middle) > 0:
            self.model.net.middle[0].register_forward_hook(hook_fn('bottleneck'))
            logger.info("Registered hook on net.middle[0]")
        else:
            logger.warning("Could not find middle blocks for hook registration")
    
    def compute_spectral_flatness(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral flatness (Wiener entropy) of a signal.
        
        Spectral flatness = geometric_mean(PSD) / arithmetic_mean(PSD)
        
        Low flatness (~0) indicates structural/tonal content (likely OOD residual)
        High flatness (~1) indicates white noise (likely ID residual)
        
        Args:
            signal: Complex signal [batch, length]
            
        Returns:
            Spectral flatness values [batch]
        """
        batch_size = signal.shape[0]
        signal_np = signal.cpu().numpy()
        
        flatness_values = []
        
        for i in range(batch_size):
            # Compute Power Spectral Density using Welch's method
            freqs, psd = scipy_signal.welch(
                signal_np[i],
                fs=1.0,  # Normalized frequency
                nperseg=min(64, len(signal_np[i])),
                scaling='density'
            )
            
            # Add small epsilon to avoid log(0)
            psd = psd + 1e-12
            
            # Spectral flatness = exp(mean(log(PSD))) / mean(PSD)
            geometric_mean = np.exp(np.mean(np.log(psd)))
            arithmetic_mean = np.mean(psd)
            
            flatness = geometric_mean / arithmetic_mean
            flatness_values.append(flatness)
        
        return torch.tensor(flatness_values, dtype=torch.float32)
    
    def compute_residual_spectrum_score(
        self,
        noisy_input: torch.Tensor,
        reconstructed_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Method A: Signal Domain Residual Spectrum Analysis
        
        Computes OOD score based on spectral characteristics of the residual.
        For ID data, residual should be white noise (high spectral flatness).
        For OOD data, residual contains structural artifacts (low spectral flatness).
        
        Args:
            noisy_input: Noisy/masked input signal [batch, 2, length]
            reconstructed_output: Reconstructed clean output [batch, 2, length]
            
        Returns:
            OOD scores [batch] (higher = more likely OOD)
        """
        # Compute residual
        residual = noisy_input - reconstructed_output  # [batch, 2, length]
        
        # Convert I/Q to complex signal
        residual_complex = residual[:, 0, :] + 1j * residual[:, 1, :]  # [batch, length]
        
        # Compute spectral flatness
        spectral_flatness = self.compute_spectral_flatness(residual_complex)  # [batch]
        
        # OOD score: negative log of spectral flatness
        # High flatness (white noise, ID) -> low score
        # Low flatness (structural, OOD) -> high score
        ood_scores = -torch.log(spectral_flatness + 1e-10)
        
        return ood_scores
    
    def extract_bottleneck_features(
        self,
        signal: torch.Tensor,
        class_labels: torch.Tensor,
        num_steps: int = 1
    ) -> torch.Tensor:
        """
        Extract bottleneck features from a signal by passing it through the encoder.
        
        Args:
            signal: Input signal [batch, 2, length]
            class_labels: Class labels for conditional denoising [batch]
            num_steps: Number of denoising steps
            
        Returns:
            Bottleneck features [batch, channels, bottleneck_length]
        """
        # Clear previous features
        self.bottleneck_features = {}
        
        # Forward pass through the model (this triggers the hook)
        with torch.no_grad():
            _ = self.model.denoise(
                x_noisy=signal,
                class_labels=class_labels,
                num_steps=num_steps
            )
        
        # Extract the bottleneck features
        if 'bottleneck' in self.bottleneck_features:
            return self.bottleneck_features['bottleneck']
        else:
            raise RuntimeError("Bottleneck features were not captured by the hook")
    
    def compute_feature_domain_distance(
        self,
        noisy_input: torch.Tensor,
        reconstructed_output: torch.Tensor,
        class_labels: torch.Tensor,
        num_steps: int = 1
    ) -> torch.Tensor:
        """
        Method B: Feature Domain Distance (Bottleneck Consistency)
        
        Compares the deep feature representations of input and output.
        If the model hallucinates (OOD), the features will be very different.
        If the model correctly denoises (ID), the features will be similar.
        
        Args:
            noisy_input: Noisy/masked input signal [batch, 2, length]
            reconstructed_output: Reconstructed clean output [batch, 2, length]
            class_labels: Class labels used for denoising [batch]
            num_steps: Number of denoising steps
            
        Returns:
            OOD scores [batch] (higher = more likely OOD)
        """
        # Extract features from input
        features_input = self.extract_bottleneck_features(noisy_input, class_labels, num_steps)
        
        # Extract features from output
        features_output = self.extract_bottleneck_features(reconstructed_output, class_labels, num_steps)
        
        # Flatten features for cosine similarity computation
        batch_size = features_input.shape[0]
        feat_in_flat = features_input.view(batch_size, -1)
        feat_out_flat = features_output.view(batch_size, -1)
        
        # Normalize features
        feat_in_norm = F.normalize(feat_in_flat, p=2, dim=1)
        feat_out_norm = F.normalize(feat_out_flat, p=2, dim=1)
        
        # Compute cosine similarity
        cosine_similarity = (feat_in_norm * feat_out_norm).sum(dim=1)  # [batch]
        
        # Cosine distance = 1 - cosine_similarity
        cosine_distance = 1.0 - cosine_similarity
        
        return cosine_distance


def parse_args():
    parser = argparse.ArgumentParser(
        description='Advanced OOD Detection Evaluation for Denoising Networks'
    )
    
    # Model and data
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained denoising model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to denoising dataset')
    parser.add_argument('--experiment_setting', type=int, default=1, 
                       choices=range(1, 13),
                       help='Experiment setting (1-12)')
    
    # Method selection
    parser.add_argument('--method', type=str, required=True,
                       choices=['residual_spectrum', 'feature_distance', 'both'],
                       help='OOD detection method')
    
    # Hyperparameters
    parser.add_argument('--num_steps', type=int, default=1,
                       help='Number of denoising steps (1-20)')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # SNR range
    parser.add_argument('--snr_min', type=int, default=-20,
                       help='Minimum SNR to evaluate')
    parser.add_argument('--snr_max', type=int, default=20,
                       help='Maximum SNR to evaluate')
    
    # Device
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--save_scores', action='store_true',
                       help='Save OOD scores to file')
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> MeanFlowDenoising:
    """Load trained denoising model from checkpoint"""
    logger.info(f'Loading model from {checkpoint_path}')
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


def compute_ood_scores_batch(
    detector: AdvancedOODDetector,
    noisy_signal: torch.Tensor,
    num_classes: int,
    method: str,
    num_steps: int,
    device: str
) -> torch.Tensor:
    """
    Compute OOD scores for a batch using the specified method.
    Tests all known classes and returns minimum score (best case).
    
    Args:
        detector: AdvancedOODDetector instance
        noisy_signal: Noisy signal [batch, 2, 128]
        num_classes: Number of known classes
        method: 'residual_spectrum' or 'feature_distance'
        num_steps: Number of denoising steps
        device: Device to use
        
    Returns:
        OOD scores [batch]
    """
    batch_size = noisy_signal.shape[0]
    min_scores = torch.full((batch_size,), float('inf'), device=device)
    
    for class_id in range(num_classes):
        class_labels = torch.full((batch_size,), class_id, dtype=torch.long, device=device)
        
        # Denoise
        with torch.no_grad():
            denoised = detector.model.denoise(
                x_noisy=noisy_signal,
                class_labels=class_labels,
                num_steps=num_steps
            )
        
        # Compute scores based on method
        if method == 'residual_spectrum':
            scores = detector.compute_residual_spectrum_score(noisy_signal, denoised)
        elif method == 'feature_distance':
            scores = detector.compute_feature_domain_distance(
                noisy_signal, denoised, class_labels, num_steps
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        scores = scores.to(device)
        min_scores = torch.minimum(min_scores, scores)
    
    return min_scores


def evaluate_ood_detection(
    model: MeanFlowDenoising,
    test_loader: DataLoader,
    method: str,
    num_steps: int,
    device: str
) -> Dict:
    """
    Evaluate OOD detection performance
    
    Args:
        model: Trained denoising model
        test_loader: Test data loader
        method: OOD detection method
        num_steps: Number of denoising steps
        device: Device to use
        
    Returns:
        Dictionary containing metrics and scores
    """
    model.eval()
    
    # Initialize detector
    logger.info(f'Initializing AdvancedOODDetector...')
    detector = AdvancedOODDetector(model)
    
    all_ood_scores = []
    all_labels = []
    all_snrs = []
    all_modulations = []
    
    logger.info(f'Computing OOD scores using method: {method}')
    logger.info(f'Number of denoising steps: {num_steps}')
    
    with torch.no_grad():
        for noisy_samples, clean_samples, labels, info in tqdm(test_loader, desc='Evaluating'):
            noisy_samples = noisy_samples.to(device)
            
            # Compute OOD scores
            ood_scores = compute_ood_scores_batch(
                detector=detector,
                noisy_signal=noisy_samples,
                num_classes=model.num_classes,
                method=method,
                num_steps=num_steps,
                device=device
            )
            
            # Convert labels: known (>=0) -> 0, unknown (-1) -> 1
            is_unknown = (labels == -1).long()
            
            all_ood_scores.append(ood_scores.cpu().numpy())
            all_labels.append(is_unknown.numpy())
            all_snrs.append(info['snr'].numpy())
            all_modulations.extend(info['original_modulation'])
    
    # Concatenate results
    all_ood_scores = np.concatenate(all_ood_scores)
    all_labels = np.concatenate(all_labels)
    all_snrs = np.concatenate(all_snrs)
    all_modulations = np.array(all_modulations)
    
    # Compute metrics
    auroc = roc_auc_score(all_labels, all_ood_scores)
    aupr = average_precision_score(all_labels, all_ood_scores)
    
    # Compute FPR@95
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
        if snr_mask.sum() > 0 and all_labels[snr_mask].sum() > 0:
            try:
                snr_auroc = roc_auc_score(all_labels[snr_mask], all_ood_scores[snr_mask])
                metrics[f'auroc_snr_{int(snr)}'] = snr_auroc
            except:
                pass
    
    # Per-modulation scores
    unique_mods = np.unique(all_modulations)
    for mod in unique_mods:
        mod_mask = all_modulations == mod
        if mod_mask.sum() > 0:
            metrics[f'score_{mod}'] = all_ood_scores[mod_mask].mean()
    
    return {
        'metrics': metrics,
        'scores': all_ood_scores,
        'labels': all_labels,
        'snrs': all_snrs,
        'modulations': all_modulations
    }


def print_results(results: Dict, known_classes: list, unknown_classes: list):
    """Print evaluation results in a clean format"""
    metrics = results['metrics']
    
    logger.info('\n' + '='*70)
    logger.info('ADVANCED OOD DETECTION RESULTS')
    logger.info('='*70)
    
    # Overall metrics
    logger.info(f'\nOverall Performance:')
    logger.info(f'  AUROC:    {metrics["auroc"]:.4f}')
    logger.info(f'  AUPR:     {metrics["aupr"]:.4f}')
    logger.info(f'  FPR@95:   {metrics["fpr@95"]:.4f}')
    
    # Per-SNR AUROC
    logger.info(f'\nPer-SNR AUROC:')
    snr_keys = sorted([k for k in metrics.keys() if k.startswith('auroc_snr_')])
    for key in snr_keys:
        snr = key.split('_')[-1]
        logger.info(f'  SNR {snr:>3s} dB: {metrics[key]:.4f}')
    
    # Per-modulation scores
    logger.info(f'\nMean OOD Scores by Modulation:')
    
    logger.info(f'  Known Classes (should have LOW scores):')
    for mod in sorted(known_classes):
        key = f'score_{mod}'
        if key in metrics:
            logger.info(f'    {mod:10s}: {metrics[key]:.4f}')
    
    logger.info(f'  Unknown Classes (should have HIGH scores):')
    for mod in sorted(unknown_classes):
        key = f'score_{mod}'
        if key in metrics:
            logger.info(f'    {mod:10s}: {metrics[key]:.4f}')
    
    logger.info('='*70 + '\n')


def save_results(results: Dict, output_dir: Path, method: str, num_steps: int):
    """Save evaluation results to file"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_file = output_dir / f'metrics_{method}_steps{num_steps}.txt'
    with open(metrics_file, 'w') as f:
        for key, value in results['metrics'].items():
            f.write(f'{key}: {value}\n')
    logger.info(f'Saved metrics to {metrics_file}')
    
    # Save scores
    scores_file = output_dir / f'scores_{method}_steps{num_steps}.npz'
    np.savez(
        scores_file,
        scores=results['scores'],
        labels=results['labels'],
        snrs=results['snrs'],
        modulations=results['modulations']
    )
    logger.info(f'Saved scores to {scores_file}')


def main():
    args = parse_args()
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    num_classes = model.num_classes
    
    # Get class information
    known_classes = EXPERIMENT_SETTINGS[args.experiment_setting]['known']
    unknown_classes = EXPERIMENT_SETTINGS[args.experiment_setting]['unknown']
    
    logger.info(f'\nExperiment Setting: {args.experiment_setting}')
    logger.info(f'Known classes ({len(known_classes)}): {known_classes}')
    logger.info(f'Unknown classes ({len(unknown_classes)}): {unknown_classes}')
    
    # Load data
    logger.info(f'\nLoading data from {args.data_path}')
    _, _, test_loader = get_rml_denoising_dataloaders(
        data_path=args.data_path,
        experiment_setting=args.experiment_setting,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        snr_range=(args.snr_min, args.snr_max),
        seed=42
    )
    
    # Evaluate
    if args.method == 'both':
        methods = ['residual_spectrum', 'feature_distance']
    else:
        methods = [args.method]
    
    for method in methods:
        logger.info(f'\n{"="*70}')
        logger.info(f'Evaluating method: {method}')
        logger.info(f'{"="*70}')
        
        results = evaluate_ood_detection(
            model=model,
            test_loader=test_loader,
            method=method,
            num_steps=args.num_steps,
            device=args.device
        )
        
        # Print results
        print_results(results, known_classes, unknown_classes)
        
        # Save results
        if args.save_scores:
            output_dir = Path(args.output_dir)
            save_results(results, output_dir, method, args.num_steps)


if __name__ == '__main__':
    main()

