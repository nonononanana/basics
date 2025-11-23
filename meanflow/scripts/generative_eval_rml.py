"""
Generative Evaluation Script for MeanFlow on RML2016.10a Dataset
Evaluates the generative quality of the trained model by comparing
generated samples with real data using multiple metrics.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats
from scipy import linalg as scipy_linalg
from scipy.spatial.distance import jensenshannon
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


def _json_default(obj):
    """Helper to make numpy scalars/arrays JSON serializable."""
    import numpy as np

    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

# Add parent directory to path for imports
script_dir = Path(__file__).parent
meanflow_dir = script_dir.parent  # meanflow/ directory
project_root = meanflow_dir.parent  # project root
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(meanflow_dir))  # Also add meanflow/ to path for relative imports

# Import with proper path handling
from meanflow.data.rml_dataset import RML2016Dataset, EXPERIMENT_SETTINGS
from meanflow.models.meanflow_modulation import MeanFlowModulation
from meanflow.models.unet_modulation import ModulationUNet

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_model(args, snr_range=(-20, 20)) -> MeanFlowModulation:
    """
    Create the Mean Flow model for modulation classification
    
    Args:
        args: Command line arguments
        snr_range: Tuple of (min_snr, max_snr) from dataset
    
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
        'class_embed_dim': None,  # Will default to model_channels * 2
        'use_snr_conditioning': True,
        'snr_range': snr_range  # Pass actual SNR range from data
    }
    
    # Create Mean Flow model
    model = MeanFlowModulation(
        arch=ModulationUNet,
        args=args,
        net_configs=net_configs,
        num_classes=num_classes,
        use_arcface=getattr(args, 'use_arcface', False),
        arcface_margin=getattr(args, 'arcface_margin', 0.3),
        arcface_scale=getattr(args, 'arcface_scale', 30.0),
        energy_temperature=getattr(args, 'energy_temperature', 1.0)
    )
    
    return model


def load_checkpoint(checkpoint_path: str, device: torch.device, snr_range=(-20, 20)) -> Tuple[nn.Module, object]:
    """
    Load checkpoint and create model
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        snr_range: Tuple of (min_snr, max_snr) from dataset
    
    Returns:
        model: Loaded model in eval mode
        args: Arguments from checkpoint
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get args from checkpoint
    args = checkpoint.get('args', None)
    if args is None:
        raise ValueError("Checkpoint does not contain 'args'. Cannot recreate model.")
    
    # Create model using saved args and actual SNR range
    model = create_model(args, snr_range=snr_range)
    model.to(device)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info("Loaded model state dict")
    else:
        logger.warning("No 'model_state_dict' found, trying 'model' key")
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            raise ValueError("Cannot find model state dict in checkpoint")
    
    # Load EMA network if available
    if 'ema_state_dict' in checkpoint:
        model.net_ema.load_state_dict(checkpoint['ema_state_dict'], strict=False)
        logger.info("Loaded EMA network state dict")
    
    # Load additional EMA networks
    if hasattr(model, 'ema_decays'):
        for i in range(len(model.ema_decays)):
            ema_key = f'ema{i + 1}_state_dict'
            if ema_key in checkpoint:
                ema_net = getattr(model, f"net_ema{i + 1}")
                ema_net.load_state_dict(checkpoint[ema_key], strict=False)
                logger.info(f"Loaded EMA{i + 1} network state dict")
    
    # Set to eval mode
    model.eval()
    
    logger.info(f"Model loaded successfully. Experiment setting: {args.experiment_setting}")
    logger.info(f"Known classes: {EXPERIMENT_SETTINGS[args.experiment_setting]['known']}")
    
    return model, args


def collect_real_samples(
    dataset: RML2016Dataset,
    class_idx: int,
    num_samples: int,
    device: torch.device,
    snr_value: Optional[float] = None
) -> torch.Tensor:
    """
    Collect real samples for a specific class and SNR
    
    Args:
        dataset: RML2016Dataset instance
        class_idx: Class index to collect samples for
        num_samples: Number of samples to collect
        device: Device to move samples to
        snr_value: SNR value in dB to filter by (optional)
    
    Returns:
        Tensor of shape [num_samples, 2, 128]
    """
    samples = []
    
    # Filter by class
    indices = np.where(dataset.labels == class_idx)[0]
    
    # Further filter by SNR if specified
    if snr_value is not None and hasattr(dataset, 'snr_labels'):
        # Get SNR labels for these indices
        snr_labels = dataset.snr_labels[indices]
        # Find samples matching the SNR value
        snr_mask = (snr_labels == snr_value)
        indices = indices[snr_mask]
    
    if len(indices) < num_samples:
        logger.warning(f"Only {len(indices)} samples available for class {class_idx}, requested {num_samples}")
        num_samples = len(indices)
    
    selected_indices = np.random.choice(indices, num_samples, replace=False)
    
    for idx in selected_indices:
        pos_sample, _, _, _ = dataset[idx]
        samples.append(pos_sample)
    
    samples_tensor = torch.stack(samples).to(device)
    return samples_tensor


def generate_samples(
    model: nn.Module,
    class_idx: int,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    snr_value: Optional[float] = None
) -> torch.Tensor:
    """
    Generate samples for a specific class and SNR
    
    Args:
        model: MeanFlowModulation model
        class_idx: Class index to generate samples for
        num_samples: Number of samples to generate
        batch_size: Batch size for generation
        device: Device to generate on
        snr_value: SNR value in dB (optional)
    
    Returns:
        Tensor of shape [num_samples, 2, 128]
    """
    model.eval()
    samples = []
    
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc=f"Generating class {class_idx}", leave=False):
            current_batch_size = min(batch_size, num_samples - i)
            class_labels = torch.full((current_batch_size,), class_idx, dtype=torch.long, device=device)
            
            # Create SNR tensor if provided
            snr_tensor = None
            if snr_value is not None:
                snr_tensor = torch.full((current_batch_size,), snr_value, dtype=torch.float32, device=device)
            
            generated = model.sample(
                samples_shape=(current_batch_size, 2, 128),
                class_labels=class_labels,
                snr_values=snr_tensor,
                device=device
            )
            samples.append(generated.cpu())
    
    samples_tensor = torch.cat(samples, dim=0)
    return samples_tensor.to(device)


def compute_mmd(x: np.ndarray, y: np.ndarray, kernel='rbf', gamma=None) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between two samples
    
    Args:
        x: First sample [N, D]
        y: Second sample [M, D]
        kernel: Kernel type ('rbf')
        gamma: RBF kernel bandwidth (auto if None)
    
    Returns:
        MMD value
    """
    if gamma is None:
        # Use median heuristic for gamma
        xx = np.sum(x**2, axis=1).reshape(-1, 1)
        yy = np.sum(y**2, axis=1).reshape(1, -1)
        xy = np.dot(x, y.T)
        distances = xx + yy - 2 * xy
        gamma = 1.0 / np.median(distances[distances > 0])
    
    def rbf_kernel(X, Y):
        XX = np.sum(X**2, axis=1).reshape(-1, 1)
        YY = np.sum(Y**2, axis=1).reshape(1, -1)
        XY = np.dot(X, Y.T)
        return np.exp(-gamma * (XX + YY - 2 * XY))
    
    Kxx = rbf_kernel(x, x)
    Kyy = rbf_kernel(y, y)
    Kxy = rbf_kernel(x, y)
    
    mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    return max(0, mmd)


def compute_c2st_accuracy(real: np.ndarray, gen: np.ndarray, use_mlp: bool = True) -> float:
    """
    Compute Classifier Two-Sample Test (C2ST) accuracy
    
    Args:
        real: Real samples [N, D]
        gen: Generated samples [M, D]
        use_mlp: Use MLP classifier (True) or Logistic Regression (False)
    
    Returns:
        C2ST accuracy (higher = more distinguishable)
    """
    # Prepare data
    X = np.vstack([real, gen])
    y = np.hstack([np.zeros(len(real)), np.ones(len(gen))])
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train classifier
    if use_mlp:
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    else:
        clf = LogisticRegression(max_iter=1000, random_state=42)
    
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    
    return accuracy


def compute_fid_like(
    model: nn.Module,
    real: torch.Tensor,
    gen: torch.Tensor,
    device: torch.device,
    class_idx: Optional[int] = None
) -> float:
    """
    Compute Fréchet distance in model's feature space
    
    Args:
        model: MeanFlowModulation model
        real: Real samples [N, 2, 128]
        gen: Generated samples [M, 2, 128]
        device: Device
        class_idx: Class index for feature extraction (uses 0 if None)
    
    Returns:
        Fréchet distance
    """
    model.eval()
    
    if class_idx is None:
        class_idx = 0
    
    def extract_features(samples: torch.Tensor) -> np.ndarray:
        batch_size = 256
        features = []
        
        with torch.no_grad():
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i+batch_size].to(device)
                batch_size_actual = batch.shape[0]
                
                # Use t=0.5, h=0.5 for feature extraction
                t = torch.full((batch_size_actual,), 0.5, device=device)
                h = torch.full((batch_size_actual,), 0.5, device=device)
                
                # Use the provided class index for feature extraction
                class_labels = torch.full((batch_size_actual,), class_idx, dtype=torch.long, device=device)
                
                feat = model.net_ema.forward_features(
                    batch,
                    time_cond=(t, h),
                    aug_cond=None,
                    class_labels=class_labels
                )
                features.append(feat.cpu().numpy())
        
        return np.vstack(features)
    
    real_feat = extract_features(real)
    gen_feat = extract_features(gen)
    
    # Compute Fréchet distance
    mu_real = np.mean(real_feat, axis=0)
    mu_gen = np.mean(gen_feat, axis=0)
    
    sigma_real = np.cov(real_feat.T)
    sigma_gen = np.cov(gen_feat.T)
    
    # Add small epsilon for numerical stability
    eps = 1e-6
    sigma_real += eps * np.eye(sigma_real.shape[0])
    sigma_gen += eps * np.eye(sigma_gen.shape[0])
    
    # Compute trace of sqrt(sigma_real @ sigma_gen)
    # Using more stable computation
    diff = mu_real - mu_gen
    covmean = scipy_linalg.sqrtm(sigma_real @ sigma_gen)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = np.sum(diff**2) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return float(fid)


def compute_psd_distance(real: np.ndarray, gen: np.ndarray) -> float:
    """
    Compute PSD (Power Spectral Density) distance between real and generated signals
    
    Args:
        real: Real samples [N, 2, 128]
        gen: Generated samples [M, 2, 128]
    
    Returns:
        L2 distance between mean PSDs
    """
    def compute_mean_psd(samples: np.ndarray) -> np.ndarray:
        # samples: [N, 2, 128]
        psds = []
        for i in range(samples.shape[0]):
            # Compute complex signal from I/Q
            signal = samples[i, 0] + 1j * samples[i, 1]
            # Compute PSD using Welch's method
            freqs, psd = welch(signal, nperseg=64, return_onesided=False)
            psds.append(psd)
        return np.mean(psds, axis=0)
    
    psd_real = compute_mean_psd(real)
    psd_gen = compute_mean_psd(gen)
    
    # L2 distance
    distance = np.sqrt(np.sum((psd_real - psd_gen)**2))
    return distance


def compute_histogram_distance(real: np.ndarray, gen: np.ndarray) -> Dict[str, float]:
    """
    Compute histogram distances for amplitude and phase
    
    Args:
        real: Real samples [N, 2, 128]
        gen: Generated samples [M, 2, 128]
    
    Returns:
        Dictionary with 'amplitude_js' and 'phase_js' (Jensen-Shannon divergence)
    """
    def extract_amplitude_phase(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # samples: [N, 2, 128]
        amplitudes = []
        phases = []
        
        for i in range(samples.shape[0]):
            signal = samples[i, 0] + 1j * samples[i, 1]
            amp = np.abs(signal)
            phase = np.angle(signal)
            amplitudes.extend(amp)
            phases.extend(phase)
        
        return np.array(amplitudes), np.array(phases)
    
    amp_real, phase_real = extract_amplitude_phase(real)
    amp_gen, phase_gen = extract_amplitude_phase(gen)
    
    # Compute histograms
    amp_bins = np.linspace(0, max(amp_real.max(), amp_gen.max()), 50)
    phase_bins = np.linspace(-np.pi, np.pi, 50)
    
    amp_hist_real, _ = np.histogram(amp_real, bins=amp_bins, density=True)
    amp_hist_gen, _ = np.histogram(amp_gen, bins=amp_bins, density=True)
    
    phase_hist_real, _ = np.histogram(phase_real, bins=phase_bins, density=True)
    phase_hist_gen, _ = np.histogram(phase_gen, bins=phase_bins, density=True)
    
    # Normalize
    amp_hist_real = amp_hist_real / (amp_hist_real.sum() + 1e-10)
    amp_hist_gen = amp_hist_gen / (amp_hist_gen.sum() + 1e-10)
    phase_hist_real = phase_hist_real / (phase_hist_real.sum() + 1e-10)
    phase_hist_gen = phase_hist_gen / (phase_hist_gen.sum() + 1e-10)
    
    # Jensen-Shannon divergence
    amp_js = jensenshannon(amp_hist_real, amp_hist_gen)
    phase_js = jensenshannon(phase_hist_real, phase_hist_gen)
    
    return {
        'amplitude_js': float(amp_js),
        'phase_js': float(phase_js)
    }


def create_visualizations(
    real: np.ndarray,
    gen: np.ndarray,
    class_name: str,
    output_dir: Path,
    num_samples_plot: int = 100
):
    """
    Create visualization plots for a class
    
    Args:
        real: Real samples [N, 2, 128]
        gen: Generated samples [M, 2, 128]
        class_name: Name of the class
        output_dir: Output directory for plots
        num_samples_plot: Number of samples to plot
    """
    # Sample subset for plotting
    n_real = min(num_samples_plot, len(real))
    n_gen = min(num_samples_plot, len(gen))
    real_plot = real[:n_real]
    gen_plot = gen[:n_gen]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Generation Quality: {class_name}', fontsize=14)
    
    # 1. Constellation plot (I vs Q)
    ax = axes[0, 0]
    real_iq = real_plot[:, 0, :].flatten()
    real_qq = real_plot[:, 1, :].flatten()
    gen_iq = gen_plot[:, 0, :].flatten()
    gen_qq = gen_plot[:, 1, :].flatten()
    
    ax.scatter(real_iq[::10], real_qq[::10], alpha=0.3, s=1, label='Real', color='blue')
    ax.scatter(gen_iq[::10], gen_qq[::10], alpha=0.3, s=1, label='Generated', color='red')
    ax.set_xlabel('I (In-phase)')
    ax.set_ylabel('Q (Quadrature)')
    ax.set_title('Constellation Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Amplitude distribution
    ax = axes[0, 1]
    real_amp = np.abs(real_plot[:, 0, :] + 1j * real_plot[:, 1, :]).flatten()
    gen_amp = np.abs(gen_plot[:, 0, :] + 1j * gen_plot[:, 1, :]).flatten()
    
    ax.hist(real_amp, bins=50, alpha=0.5, label='Real', density=True, color='blue')
    ax.hist(gen_amp, bins=50, alpha=0.5, label='Generated', density=True, color='red')
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Density')
    ax.set_title('Amplitude Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Phase distribution
    ax = axes[1, 0]
    real_phase = np.angle(real_plot[:, 0, :] + 1j * real_plot[:, 1, :]).flatten()
    gen_phase = np.angle(gen_plot[:, 0, :] + 1j * gen_plot[:, 1, :]).flatten()
    
    ax.hist(real_phase, bins=50, alpha=0.5, label='Real', density=True, color='blue')
    ax.hist(gen_phase, bins=50, alpha=0.5, label='Generated', density=True, color='red')
    ax.set_xlabel('Phase (radians)')
    ax.set_ylabel('Density')
    ax.set_title('Phase Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. PSD comparison
    ax = axes[1, 1]
    
    def compute_mean_psd(samples):
        psds = []
        for i in range(min(50, len(samples))):
            signal = samples[i, 0] + 1j * samples[i, 1]
            freqs, psd = welch(signal, nperseg=64, return_onesided=False)
            psds.append(psd)
        return freqs, np.mean(psds, axis=0)
    
    freqs, psd_real = compute_mean_psd(real_plot)
    _, psd_gen = compute_mean_psd(gen_plot)
    
    ax.plot(freqs, psd_real, label='Real', color='blue', linewidth=2)
    ax.plot(freqs, psd_gen, label='Generated', color='red', linewidth=2, linestyle='--')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('Power Spectral Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{class_name}_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_generative_quality(
    model: nn.Module,
    dataset: RML2016Dataset,
    args,
    num_samples_per_class: int = 1000,
    batch_size: int = 256,
    device: torch.device = None,
    metrics: List[str] = None,
    create_plots: bool = True,
    output_dir: Path = None,
    snr_levels: List[float] = None
) -> Dict:
    """
    Main evaluation function with per-SNR evaluation
    
    Args:
        model: MeanFlowModulation model
        dataset: RML2016Dataset test split
        args: Training arguments
        num_samples_per_class: Number of samples to generate per class per SNR
        batch_size: Batch size for generation
        device: Device
        metrics: List of metrics to compute ['mmd', 'c2st', 'fid', 'psd', 'hist']
        create_plots: Whether to create visualization plots
        output_dir: Output directory
        snr_levels: List of SNR values to evaluate (default: [-10, 0, 10, 18])
    
    Returns:
        Dictionary with all metrics including per-SNR breakdown
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if metrics is None:
        metrics = ['mmd', 'c2st', 'fid', 'psd', 'hist']
    
    # Default SNR levels: low (-10), medium (0, 10), high (18)
    if snr_levels is None:
        snr_levels = [-10, 0, 10, 18]
    
    # Categorize SNR levels into low/medium/high
    snr_categories = {
        'low': [s for s in snr_levels if s < -5],
        'medium': [s for s in snr_levels if -5 <= s <= 15],
        'high': [s for s in snr_levels if s > 15]
    }
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"outputs/generative_eval/{timestamp}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    
    # Get class information
    experiment_setting = args.experiment_setting
    known_classes = EXPERIMENT_SETTINGS[experiment_setting]['known']
    num_classes = len(known_classes)
    
    logger.info(f"Evaluating generative quality for {num_classes} classes")
    logger.info(f"Classes: {known_classes}")
    logger.info(f"SNR levels: {snr_levels}")
    logger.info(f"Generating {num_samples_per_class} samples per class per SNR")
    
    all_metrics = {}
    per_class_metrics = {}
    per_snr_metrics = {}
    per_class_per_snr_metrics = {}
    
    # Store samples for aggregated metrics
    all_real_per_snr = {snr: [] for snr in snr_levels}
    all_gen_per_snr = {snr: [] for snr in snr_levels}
    
    # Evaluate per class and per SNR
    for class_idx in range(num_classes):
        class_name = known_classes[class_idx]
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating class {class_idx}: {class_name}")
        logger.info(f"{'='*60}")
        
        per_class_per_snr_metrics[class_name] = {}
        
        # Evaluate for each SNR level
        for snr_value in snr_levels:
            logger.info(f"\n--- SNR: {snr_value} dB ---")
            
            # Collect real samples at this SNR
            logger.info(f"Collecting real samples at SNR {snr_value} dB...")
            real_samples = collect_real_samples(dataset, class_idx, num_samples_per_class, device, snr_value=snr_value)
            real_np = real_samples.cpu().numpy()
            
            # Generate samples at this SNR
            logger.info(f"Generating samples at SNR {snr_value} dB...")
            gen_samples = generate_samples(model, class_idx, num_samples_per_class, batch_size, device, snr_value=snr_value)
            gen_np = gen_samples.cpu().numpy()
            
            # Store for SNR-aggregated metrics
            all_real_per_snr[snr_value].append(real_np)
            all_gen_per_snr[snr_value].append(gen_np)
        
            # Compute metrics for this class-SNR combination
            snr_class_metrics = {}
            
            # Flatten for MMD and C2ST
            real_flat = real_np.reshape(len(real_np), -1)
            gen_flat = gen_np.reshape(len(gen_np), -1)
            
            if 'mmd' in metrics:
                logger.info("Computing MMD...")
                mmd = compute_mmd(real_flat, gen_flat)
                snr_class_metrics['mmd'] = mmd
                logger.info(f"MMD: {mmd:.6f}")
            
            if 'c2st' in metrics:
                logger.info("Computing C2ST...")
                c2st = compute_c2st_accuracy(real_flat, gen_flat)
                snr_class_metrics['c2st_accuracy'] = c2st
                logger.info(f"C2ST Accuracy: {c2st:.4f}")
            
            if 'fid' in metrics:
                logger.info("Computing FID-like...")
                try:
                    fid = compute_fid_like(model, real_samples, gen_samples, device, class_idx=class_idx)
                    snr_class_metrics['fid'] = fid
                    logger.info(f"FID-like: {fid:.6f}")
                except Exception as e:
                    logger.warning(f"FID computation failed: {e}")
            
            if 'psd' in metrics:
                logger.info("Computing PSD distance...")
                psd_dist = compute_psd_distance(real_np, gen_np)
                snr_class_metrics['psd_distance'] = psd_dist
                logger.info(f"PSD Distance: {psd_dist:.6f}")
            
            if 'hist' in metrics:
                logger.info("Computing histogram distances...")
                hist_dists = compute_histogram_distance(real_np, gen_np)
                snr_class_metrics.update(hist_dists)
                logger.info(f"Amplitude JS: {hist_dists['amplitude_js']:.6f}")
                logger.info(f"Phase JS: {hist_dists['phase_js']:.6f}")
            
            per_class_per_snr_metrics[class_name][f'snr_{snr_value}'] = snr_class_metrics
            
            # Create visualizations for key SNR levels
            if create_plots and snr_value in [-10, 0, 18]:
                logger.info("Creating visualizations...")
                create_visualizations(real_np, gen_np, f"{class_name}_snr{snr_value}", output_dir / "plots")
    
    # Compute per-SNR aggregated metrics
    logger.info(f"\n{'='*60}")
    logger.info("Computing per-SNR aggregated metrics...")
    logger.info(f"{'='*60}")
    
    for snr_value in snr_levels:
        logger.info(f"\n--- Aggregated metrics for SNR {snr_value} dB ---")
        
        snr_real_np = np.vstack(all_real_per_snr[snr_value])
        snr_gen_np = np.vstack(all_gen_per_snr[snr_value])
        snr_real_flat = snr_real_np.reshape(len(snr_real_np), -1)
        snr_gen_flat = snr_gen_np.reshape(len(snr_gen_np), -1)
        
        snr_metrics = {}
        
        if 'mmd' in metrics:
            logger.info("Computing MMD...")
            snr_mmd = compute_mmd(snr_real_flat, snr_gen_flat)
            snr_metrics['mmd'] = snr_mmd
            logger.info(f"MMD: {snr_mmd:.6f}")
        
        if 'c2st' in metrics:
            logger.info("Computing C2ST...")
            snr_c2st = compute_c2st_accuracy(snr_real_flat, snr_gen_flat)
            snr_metrics['c2st_accuracy'] = snr_c2st
            logger.info(f"C2ST Accuracy: {snr_c2st:.4f}")
        
        if 'fid' in metrics:
            logger.info("Computing FID-like...")
            try:
                snr_real_tensor = torch.from_numpy(snr_real_np).to(device)
                snr_gen_tensor = torch.from_numpy(snr_gen_np).to(device)
                snr_fid = compute_fid_like(model, snr_real_tensor, snr_gen_tensor, device, class_idx=0)
                snr_metrics['fid'] = snr_fid
                logger.info(f"FID-like: {snr_fid:.6f}")
            except Exception as e:
                logger.warning(f"FID computation failed: {e}")
        
        if 'psd' in metrics:
            logger.info("Computing PSD distance...")
            snr_psd = compute_psd_distance(snr_real_np, snr_gen_np)
            snr_metrics['psd_distance'] = snr_psd
            logger.info(f"PSD Distance: {snr_psd:.6f}")
        
        if 'hist' in metrics:
            logger.info("Computing histogram distances...")
            snr_hist = compute_histogram_distance(snr_real_np, snr_gen_np)
            snr_metrics.update(snr_hist)
            logger.info(f"Amplitude JS: {snr_hist['amplitude_js']:.6f}")
            logger.info(f"Phase JS: {snr_hist['phase_js']:.6f}")
        
        per_snr_metrics[f'snr_{snr_value}'] = snr_metrics
    
    # Compute category-level metrics (low/medium/high SNR)
    logger.info(f"\n{'='*60}")
    logger.info("Computing SNR category metrics...")
    logger.info(f"{'='*60}")
    
    category_metrics = {}
    for category, snr_list in snr_categories.items():
        if not snr_list:
            continue
        
        logger.info(f"\n--- Category: {category.upper()} SNR ({snr_list}) ---")
        
        # Aggregate samples from all SNRs in this category
        cat_real_list = []
        cat_gen_list = []
        for snr in snr_list:
            cat_real_list.extend(all_real_per_snr[snr])
            cat_gen_list.extend(all_gen_per_snr[snr])
        
        cat_real_np = np.vstack(cat_real_list)
        cat_gen_np = np.vstack(cat_gen_list)
        cat_real_flat = cat_real_np.reshape(len(cat_real_np), -1)
        cat_gen_flat = cat_gen_np.reshape(len(cat_gen_np), -1)
        
        cat_metrics = {}
        
        if 'mmd' in metrics:
            logger.info("Computing MMD...")
            cat_mmd = compute_mmd(cat_real_flat, cat_gen_flat)
            cat_metrics['mmd'] = cat_mmd
            logger.info(f"MMD: {cat_mmd:.6f}")
        
        if 'c2st' in metrics:
            logger.info("Computing C2ST...")
            cat_c2st = compute_c2st_accuracy(cat_real_flat, cat_gen_flat)
            cat_metrics['c2st_accuracy'] = cat_c2st
            logger.info(f"C2ST Accuracy: {cat_c2st:.4f}")
        
        if 'fid' in metrics:
            logger.info("Computing FID-like...")
            try:
                cat_real_tensor = torch.from_numpy(cat_real_np).to(device)
                cat_gen_tensor = torch.from_numpy(cat_gen_np).to(device)
                cat_fid = compute_fid_like(model, cat_real_tensor, cat_gen_tensor, device, class_idx=0)
                cat_metrics['fid'] = cat_fid
                logger.info(f"FID-like: {cat_fid:.6f}")
            except Exception as e:
                logger.warning(f"FID computation failed: {e}")
        
        if 'psd' in metrics:
            logger.info("Computing PSD distance...")
            cat_psd = compute_psd_distance(cat_real_np, cat_gen_np)
            cat_metrics['psd_distance'] = cat_psd
            logger.info(f"PSD Distance: {cat_psd:.6f}")
        
        if 'hist' in metrics:
            logger.info("Computing histogram distances...")
            cat_hist = compute_histogram_distance(cat_real_np, cat_gen_np)
            cat_metrics.update(cat_hist)
            logger.info(f"Amplitude JS: {cat_hist['amplitude_js']:.6f}")
            logger.info(f"Phase JS: {cat_hist['phase_js']:.6f}")
        
        category_metrics[category] = cat_metrics
    
    # Compile results
    all_metrics = {
        'experiment_setting': experiment_setting,
        'known_classes': known_classes,
        'num_samples_per_class': num_samples_per_class,
        'snr_levels': snr_levels,
        'snr_categories': snr_categories,
        'per_class_per_snr_metrics': per_class_per_snr_metrics,
        'per_snr_aggregated_metrics': per_snr_metrics,
        'category_metrics': category_metrics
    }
    
    # Save results
    results_file = output_dir / 'metrics.json'
    with open(results_file, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=_json_default)
    logger.info(f"\nResults saved to {results_file}")
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    
    logger.info("\n=== SNR Category Metrics (High/Medium/Low) ===")
    for category in ['low', 'medium', 'high']:
        if category in category_metrics:
            logger.info(f"\n{category.upper()} SNR:")
            for metric_name, value in category_metrics[category].items():
                logger.info(f"  {metric_name}: {value:.6f}")
    
    logger.info("\n=== Per-SNR Aggregated Metrics ===")
    for snr_key, metrics_dict in per_snr_metrics.items():
        logger.info(f"\n{snr_key}:")
        for metric_name, value in metrics_dict.items():
            logger.info(f"  {metric_name}: {value:.6f}")
    
    logger.info("\n=== Per-Class Per-SNR Metrics ===")
    for class_name, snr_dict in per_class_per_snr_metrics.items():
        logger.info(f"\n{class_name}:")
        for snr_key, metrics_dict in snr_dict.items():
            logger.info(f"  {snr_key}:")
            for metric_name, value in metrics_dict.items():
                logger.info(f"    {metric_name}: {value:.6f}")
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate generative quality of MeanFlow model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--data_path', type=str, default='data/RML2016.10a_dict.pkl',
                       help='Path to RML2016.10a dataset')
    parser.add_argument('--experiment_setting', type=int, default=None,
                       help='Experiment setting (1-12). If None, uses checkpoint setting.')
    parser.add_argument('--normalize', action='store_true', default=False,
                       help='Normalize I/Q samples in the dataset')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu). Auto-detects if None.')
    parser.add_argument('--samples_per_class', type=int, default=1000,
                       help='Number of samples to generate per class per SNR')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for generation')
    parser.add_argument('--metrics', type=str, default='mmd,c2st,fid,psd,hist',
                       help='Comma-separated list of metrics: mmd,c2st,fid,psd,hist')
    parser.add_argument('--snr_levels', type=str, default='-10,0,10,18',
                       help='Comma-separated SNR levels in dB (default: -10,0,10,18 for low/medium/high)')
    parser.add_argument('--plots', action='store_true',
                       help='Create visualization plots')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: outputs/generative_eval/timestamp)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Determine experiment setting first
    # We need this to load the dataset and get SNR range
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    checkpoint_args = checkpoint.get('args', None)
    if checkpoint_args is None:
        raise ValueError("Checkpoint does not contain 'args'")
    
    if args.experiment_setting is None:
        experiment_setting = checkpoint_args.experiment_setting
    else:
        experiment_setting = args.experiment_setting
    
    logger.info(f"Using experiment setting: {experiment_setting}")
    
    # Load test dataset to get actual SNR range
    logger.info(f"Loading test dataset from {args.data_path}")
    test_dataset = RML2016Dataset(
        data_path=args.data_path,
        experiment_setting=experiment_setting,
        split='test',
        snr_range=(-20, 20),
        normalize=args.normalize,
        return_snr=True,
        seed=42,
        precompute_negatives=False
    )
    
    # Get actual SNR range from dataset
    if hasattr(test_dataset, 'snr_labels'):
        actual_snr_min = float(test_dataset.snr_labels.min())
        actual_snr_max = float(test_dataset.snr_labels.max())
        snr_range = (actual_snr_min, actual_snr_max)
        logger.info(f"Detected SNR range from dataset: [{actual_snr_min}, {actual_snr_max}] dB")
    else:
        snr_range = (-20, 20)
        logger.info(f"Using default SNR range: {snr_range}")
    
    # Load checkpoint with correct SNR range
    model, checkpoint_args = load_checkpoint(args.checkpoint, device, snr_range=snr_range)
    
    logger.info(f"Test dataset loaded: {len(test_dataset)} samples")
    
    # Parse metrics
    metrics_list = [m.strip() for m in args.metrics.split(',')]
    
    # Parse SNR levels
    snr_levels = [float(s.strip()) for s in args.snr_levels.split(',')]
    logger.info(f"Evaluating at SNR levels: {snr_levels}")
    
    # Run evaluation
    results = evaluate_generative_quality(
        model=model,
        dataset=test_dataset,
        args=checkpoint_args,
        num_samples_per_class=args.samples_per_class,
        batch_size=args.batch_size,
        device=device,
        metrics=metrics_list,
        create_plots=args.plots,
        output_dir=args.output_dir,
        snr_levels=snr_levels
    )
    
    logger.info("\nEvaluation completed!")


if __name__ == '__main__':
    main()

