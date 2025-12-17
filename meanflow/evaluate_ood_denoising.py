"""
OOD Detection Evaluation using Denoising Network
Uses reconstruction error as OOD score
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from scipy.stats import kurtosis, gmean
from scipy import signal as scipy_signal
import matplotlib.pyplot as plt
import matplotlib

from meanflow.data.rml_dataset import get_rml_denoising_dataloaders, EXPERIMENT_SETTINGS, ALL_MODULATIONS
from meanflow.models.meanflow_denoising import MeanFlowDenoising
from meanflow.models.unet_denoising import DenoisingUNet

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
        # The bottleneck is in the middle blocks of the UNet
        # We'll extract features from the first middle block output
        def hook_fn(name):
            def hook(module, input, output):
                self.bottleneck_features[name] = output.detach()
            return hook
        
        # Register hook on the first middle block
        if hasattr(self.model.net_ema, 'middle') and len(self.model.net_ema.middle) > 0:
            self.model.net_ema.middle[0].register_forward_hook(hook_fn('bottleneck'))
        elif hasattr(self.model.net, 'middle') and len(self.model.net.middle) > 0:
            self.model.net.middle[0].register_forward_hook(hook_fn('bottleneck'))
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
            # This is more robust than raw FFT for noisy signals
            freqs, psd = scipy_signal.welch(
                signal_np[i],
                fs=1.0,  # Normalized frequency
                nperseg=min(64, len(signal_np[i])),
                scaling='density'
            )
            
            # Add small epsilon to avoid log(0)
            psd = psd + 1e-12
            
            # Spectral flatness = exp(mean(log(PSD))) / mean(PSD)
            # = geometric_mean / arithmetic_mean
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
        class_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract bottleneck features from a signal by passing it through the encoder.
        
        Args:
            signal: Input signal [batch, 2, length]
            class_labels: Class labels for conditional denoising [batch]
            
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
                num_steps=1
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
        class_labels: torch.Tensor
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
            
        Returns:
            OOD scores [batch] (higher = more likely OOD)
        """
        # Extract features from input
        features_input = self.extract_bottleneck_features(noisy_input, class_labels)
        
        # Extract features from output
        features_output = self.extract_bottleneck_features(reconstructed_output, class_labels)
        
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
        # Higher distance = more different = more likely OOD
        cosine_distance = 1.0 - cosine_similarity
        
        return cosine_distance


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
                       choices=['min_error', 'avg_error', 'improvement', 'snr_improvement', 'correlation', 'mdrc', 
                                'residual_spectrum', 'feature_distance'],
                       help='OOD scoring method')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--snr_min', type=int, default=-20,
                       help='Minimum SNR to evaluate')
    parser.add_argument('--snr_max', type=int, default=20,
                       help='Maximum SNR to evaluate')
    parser.add_argument('--save_visualization', action='store_true',
                       help='Save visualization images for MDRC method (one sample per class)')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for visualization images')
    
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


def estimate_snr(signal: torch.Tensor, debug_modulation: Optional[str] = None) -> torch.Tensor:
    """
    Estimate SNR of a signal batch using M2M4 estimator.
    Assumes complex signal structure (I/Q channels).
    
    Args:
        signal: [batch, 2, length]
        debug_modulation: Optional modulation name for debugging
        
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
    
    # #region agent log - Hypothesis G: Check M2M4 estimator behavior for FSK modulations
    if debug_modulation is not None and (debug_modulation == 'GFSK' or debug_modulation == 'CPFSK'):
        import json, time as _time
        DEBUG_LOG_PATH = "/Users/Axer/Desktop/py-meanflow/.cursor/debug.log"
        try:
            with open(DEBUG_LOG_PATH, 'a') as f:
                f.write(json.dumps({
                    'location': 'evaluate_ood_denoising.py:estimate_snr', 'hypothesisId': 'G',
                    'message': 'M2M4 estimator for FSK modulation',
                    'data': {
                        'modulation': debug_modulation,
                        'm2_mean': m2.mean().item(), 'm4_mean': m4.mean().item(),
                        's_term_mean': s_term.mean().item(), 'valid_ratio': valid_mask.float().mean().item(),
                        'snr_db_mean': snr_db.mean().item(), 'snr_db_std': snr_db.std().item(),
                    },
                    'timestamp': int(_time.time()*1000), 'sessionId': 'debug-session'
                }) + '\n')
        except: pass
    # #endregion
    
    return snr_db


def compute_ood_score_min_error(
    model: MeanFlowDenoising,
    noisy_signal: torch.Tensor,
    clean_signal: torch.Tensor,
    num_classes: int,
    modulation_names: Optional[List[str]] = None,
    known_class_names: Optional[List[str]] = None
) -> torch.Tensor:
    """
    OOD score based on maximum output SNR across all known classes.
    BLIND METHOD: Uses estimated SNR of denoised output.
    
    Intuition: For ID signals, denoising with the correct class produces
    high-quality output with high estimated SNR. For OOD signals, all class
    denoising attempts produce low-quality outputs with low SNR.
    
    Args:
        model: Trained denoising model
        noisy_signal: Noisy signal [batch, 2, 128]
        clean_signal: Clean signal [batch, 2, 128] (UNUSED - for compatibility only)
        num_classes: Number of known classes
        modulation_names: Original modulation names for debugging
        known_class_names: Known class names for debugging
    
    Returns:
        ood_scores: OOD scores [batch] (higher = more likely OOD)
    """
    batch_size = noisy_signal.shape[0]
    device = noisy_signal.device
    
    # #region agent log - Hypothesis F,G,H: Track per-class SNR estimates for GFSK/AM-SSB analysis
    import json, time as _time
    DEBUG_LOG_PATH = "/Users/Axer/Desktop/py-meanflow/.cursor/debug.log"
    def _debug_log(data):
        try:
            with open(DEBUG_LOG_PATH, 'a') as f:
                f.write(json.dumps({**data, 'timestamp': int(_time.time()*1000), 'sessionId': 'debug-session'}) + '\n')
        except: pass
    per_class_snrs = {}
    # #endregion
    
    max_snr = torch.full((batch_size,), float('-inf'), device=device)
    best_class_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
    
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
        
        # Estimate SNR of denoised output (BLIND)
        # High SNR = good signal quality = likely ID
        # Low SNR = poor quality = likely OOD
        snr = estimate_snr(denoised)
        
        # #region agent log - Hypothesis F: Track per-class SNR for GFSK/CPFSK similarity analysis
        class_name = known_class_names[class_id] if known_class_names else str(class_id)
        per_class_snrs[class_name] = snr.mean().item()
        # #endregion
        
        # Track which class gives best SNR
        better_mask = snr > max_snr
        best_class_idx[better_mask] = class_id
        
        # Track maximum SNR across all classes
        max_snr = torch.maximum(max_snr, snr)
    
    # #region agent log - Hypothesis F,G: Log per-sample best class for GFSK analysis
    if modulation_names is not None and known_class_names is not None:
        # Sample a few entries for logging
        for i in range(min(3, batch_size)):
            mod_name = modulation_names[i] if i < len(modulation_names) else 'unknown'
            best_class = known_class_names[best_class_idx[i].item()] if best_class_idx[i].item() < len(known_class_names) else str(best_class_idx[i].item())
            _debug_log({
                'location': 'evaluate_ood_denoising.py:compute_ood_score_min_error', 'hypothesisId': 'F_G',
                'message': 'Per-sample OOD analysis',
                'data': {
                    'sample_idx': i, 'original_modulation': mod_name,
                    'best_matching_class': best_class, 'best_snr': max_snr[i].item(),
                    'per_class_snrs': per_class_snrs,
                    'is_GFSK': mod_name == 'GFSK', 'is_AM_SSB': mod_name == 'AM-SSB',
                }
            })
    # #endregion
    
    # OOD score: negative SNR (high SNR = low OOD score)
    ood_scores = -max_snr
    
    return ood_scores


def compute_ood_score_avg_error(
    model: MeanFlowDenoising,
    noisy_signal: torch.Tensor,
    clean_signal: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    OOD score based on average output SNR across all known classes.
    BLIND METHOD: Average the estimated SNR of all class-conditional denoising outputs.
    
    Intuition: For ID signals, the average SNR across all classes should be
    reasonably high. For OOD signals, all denoising attempts fail, resulting
    in low average SNR.
    
    Args:
        model: Trained denoising model
        noisy_signal: Noisy signal [batch, 2, 128]
        clean_signal: Clean signal [batch, 2, 128] (UNUSED - for compatibility only)
        num_classes: Number of known classes
    
    Returns:
        ood_scores: OOD scores [batch] (higher = more likely OOD)
    """
    batch_size = noisy_signal.shape[0]
    device = noisy_signal.device
    
    snr_sum = torch.zeros(batch_size, device=device)
    
    # Average SNR over all known classes
    for class_id in range(num_classes):
        class_labels = torch.full((batch_size,), class_id, dtype=torch.long, device=device)
        
        with torch.no_grad():
            denoised = model.denoise(
                x_noisy=noisy_signal,
                class_labels=class_labels,
                num_steps=1
            )
        
        # Estimate SNR of denoised output (BLIND)
        snr = estimate_snr(denoised)
        snr_sum += snr
    
    snr_avg = snr_sum / num_classes
    
    # OOD score: negative average SNR (high SNR = low OOD score)
    ood_scores = -snr_avg
    
    return ood_scores


def compute_ood_score_improvement(
    model: MeanFlowDenoising,
    noisy_signal: torch.Tensor,
    clean_signal: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    OOD score based on denoising consistency across classes.
    BLIND METHOD: Measures variance of outputs across different class labels.
    
    Intuition: For ID signals, different class denoising attempts should produce
    similar outputs (consistency). For OOD signals, the model is confused and
    different classes produce very different (inconsistent) outputs.
    
    Args:
        model: Trained denoising model
        noisy_signal: Noisy signal [batch, 2, 128]
        clean_signal: Clean signal [batch, 2, 128] (UNUSED - for compatibility only)
        num_classes: Number of known classes
    
    Returns:
        ood_scores: OOD scores [batch] (higher = more likely OOD)
    """
    batch_size = noisy_signal.shape[0]
    device = noisy_signal.device
    
    # Collect all denoised outputs
    all_denoised = []
    
    for class_id in range(num_classes):
        class_labels = torch.full((batch_size,), class_id, dtype=torch.long, device=device)
        
        with torch.no_grad():
            denoised = model.denoise(
                x_noisy=noisy_signal,
                class_labels=class_labels,
                num_steps=1
            )
        
        all_denoised.append(denoised)
    
    # Stack: [num_classes, batch, 2, 128]
    all_denoised = torch.stack(all_denoised, dim=0)
    
    # Compute variance across classes (BLIND)
    # High variance = inconsistent = OOD
    # Low variance = consistent = ID
    variance = torch.var(all_denoised, dim=0)  # [batch, 2, 128]
    inconsistency = variance.mean(dim=(1, 2))  # [batch]
    
    # OOD score: higher inconsistency = higher OOD likelihood
    ood_scores = inconsistency
    
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


def compute_ood_score_correlation(
    model: MeanFlowDenoising,
    noisy_signal: torch.Tensor,
    clean_signal: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    OOD score based on Input-Output Correlation.
    ID samples should have higher correlation with their denoised versions.
    OOD samples (if hallucinated) will have low correlation with input.
    
    Score = -Max_Correlation (Lower correlation -> Higher OOD score)
    
    Args:
        model: Trained denoising model
        noisy_signal: Noisy signal [batch, 2, 128]
        clean_signal: Clean signal (Unused for blind score)
        num_classes: Number of known classes
    
    Returns:
        ood_scores: OOD scores [batch]
    """
    batch_size = noisy_signal.shape[0]
    device = noisy_signal.device
    
    # Normalize noisy signal for correlation calculation
    # [batch, 2, 128] -> [batch, 256]
    noisy_flat = noisy_signal.view(batch_size, -1)
    # Subtract mean
    noisy_flat = noisy_flat - noisy_flat.mean(dim=1, keepdim=True)
    # Normalize
    noisy_norm = torch.norm(noisy_flat, dim=1, keepdim=True) + 1e-8
    noisy_flat = noisy_flat / noisy_norm
    
    max_corr = torch.full((batch_size,), -1.0, device=device)
    
    for class_id in range(num_classes):
        class_labels = torch.full((batch_size,), class_id, dtype=torch.long, device=device)
        
        with torch.no_grad():
            denoised = model.denoise(
                x_noisy=noisy_signal,
                class_labels=class_labels,
                num_steps=20
            )
        
        # Calculate correlation
        denoised_flat = denoised.view(batch_size, -1)
        denoised_flat = denoised_flat - denoised_flat.mean(dim=1, keepdim=True)
        denoised_norm = torch.norm(denoised_flat, dim=1, keepdim=True) + 1e-8
        denoised_flat = denoised_flat / denoised_norm
        
        # Dot product of normalized vectors = Cosine Similarity = Correlation
        # (since means are 0)
        corr = (noisy_flat * denoised_flat).sum(dim=1)
        
        max_corr = torch.maximum(max_corr, corr)
    
    # High correlation = ID (low score)
    # Low correlation = OOD (high score)
    return -max_corr


def save_mdrc_visualization(
    noisy_signal: torch.Tensor,
    denoised_signal: torch.Tensor,
    class_name: str,
    output_dir: Path
) -> None:
    """
    Save visualization of noisy, denoised, and residual signals for MDRC analysis.
    
    Args:
        noisy_signal: Noisy signal [2, length] (I/Q channels)
        denoised_signal: Denoised signal [2, length]
        class_name: Class name for labeling
        output_dir: Directory to save images
    """
    # Convert to numpy and move to CPU
    noisy_np = noisy_signal.cpu().numpy()  # [2, length]
    denoised_np = denoised_signal.cpu().numpy()  # [2, length]
    residual_np = noisy_np - denoised_np  # [2, length]
    
    # Convert to complex for better visualization
    noisy_complex = noisy_np[0] + 1j * noisy_np[1]
    denoised_complex = denoised_np[0] + 1j * denoised_np[1]
    residual_complex = residual_np[0] + 1j * residual_np[1]
    
    # Set matplotlib to use non-interactive backend
    matplotlib.use('Agg')
    
    # 1. Plot noisy signal
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(noisy_np[0], label='I channel', alpha=0.7)
    ax1.plot(noisy_np[1], label='Q channel', alpha=0.7)
    ax1.set_title(f'{class_name}: Noisy Signal (Time Domain)')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(noisy_np[0], noisy_np[1], alpha=0.5, s=10)
    ax2.set_title(f'{class_name}: Noisy Signal (I/Q Constellation)')
    ax2.set_xlabel('I')
    ax2.set_ylabel('Q')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{class_name}_noisy.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # 2. Plot denoised signal
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(denoised_np[0], label='I channel', alpha=0.7)
    ax1.plot(denoised_np[1], label='Q channel', alpha=0.7)
    ax1.set_title(f'{class_name}: Denoised Signal (Time Domain)')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(denoised_np[0], denoised_np[1], alpha=0.5, s=10)
    ax2.set_title(f'{class_name}: Denoised Signal (I/Q Constellation)')
    ax2.set_xlabel('I')
    ax2.set_ylabel('Q')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{class_name}_denoised.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # 3. Plot residual signal
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Time domain
    ax1.plot(residual_np[0], label='I channel', alpha=0.7)
    ax1.plot(residual_np[1], label='Q channel', alpha=0.7)
    ax1.set_title(f'{class_name}: Residual Signal (Time Domain)')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # I/Q constellation
    ax2.scatter(residual_np[0], residual_np[1], alpha=0.5, s=10)
    ax2.set_title(f'{class_name}: Residual Signal (I/Q Constellation)')
    ax2.set_xlabel('I')
    ax2.set_ylabel('Q')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Amplitude
    residual_amp = np.abs(residual_complex)
    ax3.plot(residual_amp, color='purple', alpha=0.7)
    ax3.set_title(f'{class_name}: Residual Amplitude')
    ax3.set_xlabel('Sample')
    ax3.set_ylabel('Amplitude')
    ax3.grid(True, alpha=0.3)
    
    # Frequency domain
    residual_fft = np.fft.fft(residual_complex)
    residual_psd = np.abs(residual_fft)**2
    freq = np.fft.fftfreq(len(residual_complex))
    ax4.plot(np.fft.fftshift(freq), np.fft.fftshift(10 * np.log10(residual_psd + 1e-12)), 
             color='red', alpha=0.7)
    ax4.set_title(f'{class_name}: Residual Power Spectral Density')
    ax4.set_xlabel('Normalized Frequency')
    ax4.set_ylabel('Power (dB)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{class_name}_residual.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    logger.info(f'Saved visualization for {class_name} to {output_dir}')


def extract_residual_features(input_sig: torch.Tensor, denoised_sig: torch.Tensor, debug_modulations: Optional[List[str]] = None) -> np.ndarray:
    """
    Extract multi-domain features from residual signal (input - denoised).
    
    Features:
    1. Amplitude domain: Kurtosis of amplitude (for QAM, AM, PAM)
    2. Phase/Frequency domain: Variance of 2nd-order phase difference (for GFSK, CPFSK, PSK)
    3. Spectral domain: Spectral Flatness Measure (for SSB, DSB, WBFM)
    
    Args:
        input_sig: Input signal [batch, 2, length] (I/Q channels)
        denoised_sig: Denoised signal [batch, 2, length]
        debug_modulations: Optional modulation names for debugging
        
    Returns:
        features: [batch, 3] feature vector
    """
    # Compute residual
    residual = input_sig - denoised_sig  # [batch, 2, length]
    
    # Convert to complex for easier manipulation
    # residual_complex = [batch, length]
    residual_complex = residual[:, 0, :] + 1j * residual[:, 1, :]
    
    batch_size = residual_complex.shape[0]
    
    # Convert to numpy
    residual_np = residual_complex.cpu().numpy()
    
    # --- Feature 1: Amplitude domain (Kurtosis) - Vectorized ---
    res_amp = np.abs(residual_np)  # [batch, length]
    feat_amp = kurtosis(res_amp, axis=1)  # [batch]
    
    # --- Feature 2: Phase/Frequency domain - Vectorized ---
    res_phase = np.angle(residual_np)  # [batch, length]
    # Unwrap phase for each sample (unfortunately np.unwrap doesn't vectorize directly)
    res_phase_unwrapped = np.empty_like(res_phase)
    for i in range(batch_size):
        res_phase_unwrapped[i] = np.unwrap(res_phase[i])
    
    # 2nd order difference (frequency change rate)
    res_freq_change = np.diff(res_phase_unwrapped, n=2, axis=1)  # [batch, length-2]
    # Use kurtosis for better discrimination
    feat_phase = kurtosis(res_freq_change, axis=1)  # [batch]
    
    # --- Feature 3: Spectral domain (Spectral Flatness Measure) - Vectorized ---
    # Compute power spectral density
    f_res = np.fft.fft(residual_np, axis=1)  # [batch, length]
    psd = np.abs(f_res)**2 + 1e-12  # [batch, length]
    
    # Spectral Flatness Measure (SFM) = geometric_mean / arithmetic_mean
    # gmean and mean across the frequency axis (axis=1)
    sfm = gmean(psd, axis=1) / np.mean(psd, axis=1)  # [batch]
    
    # Use -log(SFM) so higher value = more OOD
    feat_spec = -np.log10(sfm + 1e-12)  # [batch]
    
    # Stack features
    features = np.stack([feat_amp, feat_phase, feat_spec], axis=1)  # [batch, 3]
    
    # #region agent log - Hypothesis H: Track residual feature distribution for GFSK vs AM-SSB
    if debug_modulations is not None:
        import json, time as _time
        DEBUG_LOG_PATH = "/Users/Axer/Desktop/py-meanflow/.cursor/debug.log"
        for i, mod in enumerate(debug_modulations[:5]):  # Log first 5 samples
            if mod in ['GFSK', 'AM-SSB', 'CPFSK']:  # Focus on these modulations
                try:
                    with open(DEBUG_LOG_PATH, 'a') as f:
                        f.write(json.dumps({
                            'location': 'evaluate_ood_denoising.py:extract_residual_features', 'hypothesisId': 'H',
                            'message': 'Residual features for OOD analysis',
                            'data': {
                                'modulation': mod, 'sample_idx': i,
                                'feat_amp_kurtosis': float(feat_amp[i]),
                                'feat_phase_kurtosis': float(feat_phase[i]),
                                'feat_spec_sfm': float(feat_spec[i]),
                                'residual_energy': float(np.mean(np.abs(residual_np[i])**2)),
                            },
                            'timestamp': int(_time.time()*1000), 'sessionId': 'debug-session'
                        }) + '\n')
                except: pass
    # #endregion
    
    return features


def compute_mahalanobis_statistics(
    model: MeanFlowDenoising,
    val_loader: DataLoader,
    device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and covariance matrix from ID validation data.
    
    For ID validation samples with known labels, we denoise using the TRUE
    class labels to obtain the residual feature distribution when the model
    is used correctly. This defines the "normal" residual pattern for ID data.
    
    Args:
        model: Trained denoising model
        val_loader: Validation data loader (must contain ID samples with labels)
        device: Device to use
        
    Returns:
        mu: Mean feature vector [3] from ID residuals
        cov_inv: Inverse covariance matrix [3, 3] from ID residuals
    """
    model.eval()
    all_features = []
    num_classes = model.num_classes
    
    logger.info('Computing Mahalanobis statistics from ID validation data...')
    
    with torch.no_grad():
        for noisy_samples, clean_samples, labels, info in tqdm(val_loader, desc='Computing statistics'):
            noisy_samples = noisy_samples.to(device)
            
            # Only use known class samples (ID samples with valid labels)
            known_mask = labels >= 0
            if known_mask.sum() == 0:
                continue
                
            noisy_samples = noisy_samples[known_mask]
            labels = labels[known_mask].to(device)
            
            # For ID validation samples, we DO have true labels
            # Denoise using the correct class label to get "ideal" residuals
            with torch.no_grad():
                denoised = model.denoise(
                    x_noisy=noisy_samples,
                    class_labels=labels,
                    num_steps=1
                )
            
            # Extract features from residuals when model is used correctly
            features = extract_residual_features(noisy_samples, denoised)
            all_features.append(features)
    
    # Concatenate all features
    all_features = np.concatenate(all_features, axis=0)
    
    # Compute statistics
    mu = np.mean(all_features, axis=0)
    cov = np.cov(all_features, rowvar=False)
    
    # Add regularization to ensure invertibility
    cov = cov + np.eye(cov.shape[0]) * 1e-6
    
    # Compute inverse
    cov_inv = np.linalg.inv(cov)
    
    logger.info(f'Computed statistics from {all_features.shape[0]} ID samples')
    logger.info(f'Mean: {mu}')
    logger.info(f'Covariance:\n{cov}')
    
    return mu, cov_inv


def compute_ood_score_mdrc(
    model: MeanFlowDenoising,
    noisy_signal: torch.Tensor,
    clean_signal: torch.Tensor,
    num_classes: int,
    mu: np.ndarray,
    cov_inv: np.ndarray,
    save_visualization: bool = False,
    output_dir: Optional[str] = None,
    known_class_names: Optional[List[str]] = None
) -> torch.Tensor:
    """
    OOD score based on Multi-Domain Residual Complexity (MDRC).
    BLIND METHOD: Only uses noisy input, no clean signal or true labels.
    
    Computes Mahalanobis distance using residual features from three domains:
    - Amplitude (Kurtosis of |r[n]|)
    - Phase/Frequency (Kurtosis of 2nd-order phase differences)
    - Spectral (Spectral Flatness Measure)
    
    For each test sample:
    1. Try denoising with all known class labels (blind - no true label)
    2. Extract 3D feature vector from residual for each attempt
    3. Compute Mahalanobis distance to ID distribution
    4. Return minimum distance (best case scenario)
    
    Intuition:
    - ID samples: At least one class produces "normal" residuals → low distance
    - OOD samples: All classes produce abnormal residuals → high distance
    
    Args:
        model: Trained denoising model
        noisy_signal: Noisy signal [batch, 2, 128]
        clean_signal: Clean signal [batch, 2, 128] (UNUSED - for API compatibility)
        num_classes: Number of known classes
        mu: Mean feature vector from ID data [3]
        cov_inv: Inverse covariance matrix [3, 3]
        save_visualization: If True, save visualization images for all modulation types
        output_dir: Directory to save visualization images
        known_class_names: List of known class names for visualization
    
    Returns:
        ood_scores: Mahalanobis distances [batch] (higher = more likely OOD)
    """
    batch_size = noisy_signal.shape[0]
    device = noisy_signal.device
    
    # Create output directory if visualization is requested
    if save_visualization and output_dir is not None:
        vis_dir = Path(output_dir) / 'eval_mdrc'
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Save visualization for all 11 modulation types
        if known_class_names is not None:
            logger.info(f'Saving visualization for all {len(ALL_MODULATIONS)} modulation types...')
            for mod_name in ALL_MODULATIONS:
                # Check if this modulation is a known class
                if mod_name in known_class_names:
                    # Get the class index for known classes
                    class_id = known_class_names.index(mod_name)
                    class_labels = torch.full((batch_size,), class_id, dtype=torch.long, device=device)
                    
                    # Denoise using the known class
                    with torch.no_grad():
                        denoised = model.denoise(
                            x_noisy=noisy_signal,
                            class_labels=class_labels,
                            num_steps=1
                        )
                else:
                    # For unknown classes, use the null label (-1) for unconditional denoising
                    # This allows visualizing what the model produces without forcing a specific class
                    class_labels = torch.full((batch_size,), -1, dtype=torch.long, device=device)
                    with torch.no_grad():
                        denoised = model.denoise(
                            x_noisy=noisy_signal,
                            class_labels=class_labels,
                            num_steps=1
                        )
                
                # Save visualization for the first sample
                save_mdrc_visualization(
                    noisy_signal[0],
                    denoised[0],
                    mod_name,
                    vis_dir
                )
    
    # For each sample, try all known classes and use minimum distance
    min_distances = torch.full((batch_size,), float('inf'), device=device)
    
    for class_id in range(num_classes):
        class_labels = torch.full((batch_size,), class_id, dtype=torch.long, device=device)
        
        # Denoise
        with torch.no_grad():
            denoised = model.denoise(
                x_noisy=noisy_signal,
                class_labels=class_labels,
                num_steps=1
            )
        
        # Extract residual features
        features = extract_residual_features(noisy_signal, denoised)  # [batch, 3]
        
        # Compute Mahalanobis distance for all samples (vectorized)
        delta = features - mu  # [batch_size, 3] - broadcasting
        # Efficient batch Mahalanobis: sqrt(sum((delta @ cov_inv) * delta, axis=1))
        distances = np.sqrt(np.sum((delta @ cov_inv) * delta, axis=1))  # [batch_size]
        
        # Convert to torch tensor
        distances_torch = torch.from_numpy(distances).float().to(device)
        
        # Update minimum distances
        min_distances = torch.minimum(min_distances, distances_torch)
    
    return min_distances


def compute_ood_score_residual_spectrum(
    model: MeanFlowDenoising,
    noisy_signal: torch.Tensor,
    clean_signal: torch.Tensor,
    num_classes: int,
    ood_detector: AdvancedOODDetector
) -> torch.Tensor:
    """
    Method A: Signal Domain Residual Spectrum Analysis
    
    OOD score based on spectral flatness of the residual signal.
    For ID data, residual should be white noise (high flatness).
    For OOD data, residual contains structural artifacts (low flatness).
    
    Args:
        model: Trained denoising model
        noisy_signal: Noisy signal [batch, 2, 128]
        clean_signal: Clean signal [batch, 2, 128] (UNUSED - for compatibility only)
        num_classes: Number of known classes
        ood_detector: AdvancedOODDetector instance
    
    Returns:
        ood_scores: OOD scores [batch] (higher = more likely OOD)
    """
    batch_size = noisy_signal.shape[0]
    device = noisy_signal.device
    
    # Try denoising with all known classes and take minimum score
    min_scores = torch.full((batch_size,), float('inf'), device=device)
    
    for class_id in range(num_classes):
        class_labels = torch.full((batch_size,), class_id, dtype=torch.long, device=device)
        
        # Denoise
        with torch.no_grad():
            denoised = model.denoise(
                x_noisy=noisy_signal,
                class_labels=class_labels,
                num_steps=1
            )
        
        # Compute spectral flatness-based OOD score
        scores = ood_detector.compute_residual_spectrum_score(noisy_signal, denoised)
        scores = scores.to(device)
        
        # Take minimum across all classes (best case)
        min_scores = torch.minimum(min_scores, scores)
    
    return min_scores


def compute_ood_score_feature_distance(
    model: MeanFlowDenoising,
    noisy_signal: torch.Tensor,
    clean_signal: torch.Tensor,
    num_classes: int,
    ood_detector: AdvancedOODDetector
) -> torch.Tensor:
    """
    Method B: Feature Domain Distance (Bottleneck Consistency)
    
    OOD score based on cosine distance between bottleneck features
    of input and output. Larger distance indicates the model produced
    a semantically different output (hallucination), likely OOD.
    
    Args:
        model: Trained denoising model
        noisy_signal: Noisy signal [batch, 2, 128]
        clean_signal: Clean signal [batch, 2, 128] (UNUSED - for compatibility only)
        num_classes: Number of known classes
        ood_detector: AdvancedOODDetector instance
    
    Returns:
        ood_scores: OOD scores [batch] (higher = more likely OOD)
    """
    batch_size = noisy_signal.shape[0]
    device = noisy_signal.device
    
    # Try denoising with all known classes and take minimum distance
    min_distances = torch.full((batch_size,), float('inf'), device=device)
    
    for class_id in range(num_classes):
        class_labels = torch.full((batch_size,), class_id, dtype=torch.long, device=device)
        
        # Denoise
        with torch.no_grad():
            denoised = model.denoise(
                x_noisy=noisy_signal,
                class_labels=class_labels,
                num_steps=1
            )
        
        # Compute feature domain distance
        distances = ood_detector.compute_feature_domain_distance(
            noisy_signal, denoised, class_labels
        )
        
        # Take minimum across all classes (best case)
        min_distances = torch.minimum(min_distances, distances)
    
    return min_distances


def evaluate_ood_detection(
    model: MeanFlowDenoising,
    test_loader: DataLoader,
    method: str,
    device: str,
    mu: np.ndarray = None,
    cov_inv: np.ndarray = None,
    save_visualization: bool = False,
    output_dir: Optional[str] = None,
    known_class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluate OOD detection performance
    
    Args:
        model: Trained denoising model
        test_loader: Test data loader
        method: OOD scoring method ('min_error', 'avg_error', 'improvement', 'mdrc')
        device: Device to use
        mu: Mean feature vector for MDRC method (optional)
        cov_inv: Inverse covariance matrix for MDRC method (optional)
        save_visualization: If True, save visualization images for MDRC method
        output_dir: Directory to save visualization images
        known_class_names: List of known class names for MDRC visualization
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    all_ood_scores = []
    all_labels = []  # 0=known, 1=unknown
    all_snrs = []
    all_modulations = []
    
    # Flag to save visualization only once (first batch)
    visualization_saved = False
    
    # Initialize AdvancedOODDetector for new methods
    ood_detector = None
    if method in ['residual_spectrum', 'feature_distance']:
        logger.info('Initializing AdvancedOODDetector...')
        ood_detector = AdvancedOODDetector(model)
    
    # Collect OOD scores
    logger.info(f'Computing OOD scores using method: {method}')
    
    with torch.no_grad():
        for noisy_samples, clean_samples, labels, info in tqdm(test_loader, desc='OOD Evaluation'):
            noisy_samples = noisy_samples.to(device)
            clean_samples = clean_samples.to(device)
            
            # Compute OOD scores
            if method == 'min_error':
                ood_scores = compute_ood_score_min_error(
                    model, noisy_samples, clean_samples, model.num_classes,
                    modulation_names=list(info['original_modulation']),
                    known_class_names=known_class_names
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
            elif method == 'correlation':
                ood_scores = compute_ood_score_correlation(
                    model, noisy_samples, clean_samples, model.num_classes
                )
            elif method == 'mdrc':
                if mu is None or cov_inv is None:
                    raise ValueError('MDRC method requires mu and cov_inv statistics')
                # Save visualization only for first batch
                should_visualize = save_visualization and not visualization_saved
                ood_scores = compute_ood_score_mdrc(
                    model, noisy_samples, clean_samples, model.num_classes, mu, cov_inv,
                    save_visualization=should_visualize, output_dir=output_dir,
                    known_class_names=known_class_names
                )
                if should_visualize:
                    visualization_saved = True
            elif method == 'residual_spectrum':
                if ood_detector is None:
                    raise ValueError('residual_spectrum method requires AdvancedOODDetector')
                ood_scores = compute_ood_score_residual_spectrum(
                    model, noisy_samples, clean_samples, model.num_classes, ood_detector
                )
            elif method == 'feature_distance':
                if ood_detector is None:
                    raise ValueError('feature_distance method requires AdvancedOODDetector')
                ood_scores = compute_ood_score_feature_distance(
                    model, noisy_samples, clean_samples, model.num_classes, ood_detector
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
    _, val_loader, test_loader = get_rml_denoising_dataloaders(
        data_path=args.data_path,
        experiment_setting=args.experiment_setting,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        snr_range=(args.snr_min, args.snr_max),
        seed=42
    )
    
    # Get known class names for visualization
    known_class_names = EXPERIMENT_SETTINGS[args.experiment_setting]['known']
    
    # Compute Mahalanobis statistics if using MDRC method
    mu, cov_inv = None, None
    if args.method == 'mdrc':
        mu, cov_inv = compute_mahalanobis_statistics(
            model=model,
            val_loader=val_loader,
            device=args.device
        )
    
    # Evaluate OOD detection
    metrics, ood_scores, labels, snrs, modulations = evaluate_ood_detection(
        model=model,
        test_loader=test_loader,
        method=args.method,
        device=args.device,
        mu=mu,
        cov_inv=cov_inv,
        save_visualization=args.save_visualization,
        output_dir=args.output_dir,
        known_class_names=known_class_names
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
