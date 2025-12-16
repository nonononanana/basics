"""
RML2016.10a Dataset Loader for Open-Set Modulation Classification
Implements proper train/test splits according to known/unknown class configurations
"""

from cmath import phase
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import logging
from scipy import interpolate, signal
import time

logger = logging.getLogger(__name__)


def apply_random_block_masking(signal: np.ndarray, mask_ratio: float, seed: Optional[int] = None) -> np.ndarray:
    """
    Apply random block masking with zero filling to I/Q signal.
    
    Randomly selects a starting point t_start and masks the next len_mask consecutive
    time points for both I and Q channels.
    
    Args:
        signal: I/Q signal of shape [2, 128] (channels x time)
        mask_ratio: Ratio of signal length to mask (0.0 to 1.0)
        seed: Optional random seed for reproducibility
        
    Returns:
        Masked signal with same shape as input [2, 128]
    """
    if mask_ratio <= 0.0 or mask_ratio >= 1.0:
        # No masking or invalid ratio
        return signal
    
    # Calculate mask length based on ratio
    signal_length = signal.shape[1]  # 128 time points
    len_mask = int(signal_length * mask_ratio)
    
    if len_mask == 0:
        return signal
    
    # Set random seed if provided
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    # Randomly choose start point (ensure we don't go out of bounds)
    t_start = rng.randint(0, signal_length - len_mask + 1)
    
    # Create masked copy of signal
    masked_signal = signal.copy()
    
    # Zero out the consecutive block for both I and Q channels
    masked_signal[:, t_start:t_start + len_mask] = 0.0
    
    return masked_signal


# Define the modulation classes in RML2016.10a dataset
ALL_MODULATIONS = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 
                   'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']

# Define the experimental settings from the table in the image
EXPERIMENT_SETTINGS = {
    # Setting 1: 9 known, 2 unknown
    1: {
        'known': ['8PSK', 'AM-DSB', 'BPSK', 'CPFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM'],
        'unknown': ['AM-SSB', 'GFSK']
    },
    # Setting 2: 6 known, 5 unknown  
    2: {
        'known': ['AM-DSB', 'AM-SSB', 'PAM4', 'QAM16', 'QAM64', 'WBFM'],
        'unknown': ['8PSK', 'BPSK', 'CPFSK', 'GFSK', 'QPSK']
    },
    # Setting 3: 4 known, 7 unknown
    3: {
        'known': ['AM-DSB', 'QAM16', 'QAM64', 'WBFM'],
        'unknown': ['8PSK', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QPSK']
    },
    # Setting 4: 3 known, 8 unknown
    4: {
        'known': ['QAM16', 'QAM64', 'WBFM'],
        'unknown': ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QPSK']
    },
    # Setting 5: 8 known, 2 unknown (10B)
    5: {
        'known': ['8PSK', 'AM-DSB', 'BPSK', 'GFSK', 'PAM4', 'QAM16', 'QPSK', 'WBFM'],
        'unknown': ['CPFSK', 'QAM64']
    },
    # Setting 6: 7 known, 3 unknown (10B)
    6: {
        'known': ['8PSK', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM64', 'QPSK'],
        'unknown': ['AM-DSB', 'QAM16', 'WBFM']
    },
    # Setting 7: 5 known, 5 unknown (10B)
    7: {
        'known': ['PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM'],
        'unknown': ['8PSK', 'AM-DSB', 'BPSK', 'CPFSK', 'GFSK']
    },
    # Setting 8: 4 known, 6 unknown (10B)
    8: {
        'known': ['AM-DSB', 'BPSK', 'PAM4', 'QAM64'],
        'unknown': ['8PSK', 'CPFSK', 'GFSK', 'QAM16', 'QPSK', 'WBFM']
    },
    # Setting 9: 9 known, 2 unknown (04C)
    9: {
        'known': ['AM-DSB', 'AM-SSB', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM'],
        'unknown': ['8PSK', 'BPSK']
    },
    # Setting 10: 7 known, 4 unknown (04C)
    10: {
        'known': ['AM-DSB', 'AM-SSB', 'BPSK', 'GFSK', 'QAM16', 'QAM64', 'WBFM'],
        'unknown': ['8PSK', 'CPFSK', 'PAM4', 'QPSK']
    },
    # Setting 11: 5 known, 6 unknown (04C)
    11: {
        'known': ['BPSK', 'GFSK', 'QAM16', 'QAM64', 'WBFM'],
        'unknown': ['8PSK', 'AM-DSB', 'AM-SSB', 'CPFSK', 'PAM4', 'QPSK']
    },
    # Setting 12: 3 known, 8 unknown (04C)
    12: {
        'known': ['QAM16', 'QAM64', 'WBFM'],
        'unknown': ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QPSK']
    }
}


class RML2016DenoisingDataset(Dataset):
    """
    Dataset class for RML2016 Denoising Task
    Uses 4-channel format: (N, 4, 128)
    - Channels 0,1: Noisy signal (Real, Imaginary)
    - Channels 2,3: Clean signal (Real, Imaginary)
    
    The model learns to denoise: noisy -> clean, conditioned on modulation type
    """
    
    def __init__(
        self,
        data_path: str,
        experiment_setting: int,
        split: str = 'train',  # 'train', 'val', or 'test'
        snr_range: Tuple[int, int] = (-20, 20),  # SNR range to include
        train_split: float = 0.8,  # 80% for training
        val_split: float = 0.1,  # 10% for validation
        test_split: float = 0.1,  # 10% for testing
        normalize: bool = False,  # Normalize I/Q samples
        seed: int = 42,
        mask_ratio: float = 0.0  # Ratio of signal to mask (0.0-1.0)
    ):
        """
        Initialize RML2016 Denoising Dataset
        
        Args:
            data_path: Path to denoising dataset file (4-channel format)
            experiment_setting: Which experimental setting (1-12) to use
            split: Dataset split - 'train', 'val', or 'test'
            snr_range: Range of SNR values to include
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
            test_split: Proportion of data for testing
            normalize: Whether to normalize I/Q samples
            seed: Random seed for reproducibility
            mask_ratio: Ratio of signal length to mask (0.0-1.0) for masked autoencoder training
        """
        super().__init__()
        
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}. Must be 'train', 'val', or 'test'"
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
        assert 0.0 <= mask_ratio < 1.0, f"mask_ratio must be in [0.0, 1.0), got {mask_ratio}"
        
        self.data_path = data_path
        self.experiment_setting = experiment_setting
        self.split = split
        self.snr_range = snr_range
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.normalize = normalize
        self.seed = seed
        self.mask_ratio = mask_ratio
        
        # Load experiment configuration
        assert experiment_setting in EXPERIMENT_SETTINGS, f"Invalid experiment setting: {experiment_setting}"
        self.known_classes = EXPERIMENT_SETTINGS[experiment_setting]['known']
        self.unknown_classes = EXPERIMENT_SETTINGS[experiment_setting]['unknown']
        
        # Create class to index mapping (only for known classes)
        self.class_to_idx = {mod: idx for idx, mod in enumerate(self.known_classes)}
        self.idx_to_class = {idx: mod for mod, idx in self.class_to_idx.items()}
        self.num_classes = len(self.known_classes)
        
        # Load and prepare data
        self._load_data()
        
    def _load_data(self):
        """Load and prepare denoising dataset with train/val/test splits"""
        # Load the pickle file
        with open(self.data_path, 'rb') as f:
            data_dict = pickle.load(f, encoding='latin1')
        
        # Separate data by modulation type and SNR
        self.noisy_samples = []
        self.clean_samples = []
        self.labels = []
        self.snrs = []
        self.is_unknown = []
        self.original_modulations = []
        
        # IMPORTANT: Set seed to ensure consistent splits across train/val/test datasets
        # All three datasets (train/val/test) use the same seed, ensuring that
        # np.random.permutation() produces identical shuffles for each (mod, snr) pair
        # This guarantees non-overlapping and consistent splits
        np.random.seed(self.seed)
        
        # Process each (modulation, SNR) key
        for (mod, snr), signals in data_dict.items():
            # Filter by SNR range
            if snr < self.snr_range[0] or snr > self.snr_range[1]:
                continue
            
            # Check signal shape - should be (N, 4, 128) for denoising
            if signals.shape[1] != 4:
                logger.warning(f"Skipping {mod} at SNR {snr}: expected 4 channels, got {signals.shape[1]}")
                continue
                
            # Determine if this is known or unknown class
            if mod in self.known_classes:
                label = self.class_to_idx[mod]
                is_unk = False
            elif mod in self.unknown_classes:
                label = -1  # Unknown class label
                is_unk = True
            else:
                logger.info(f"Error: {mod} is outer data from current dataset")
                continue
            
            # Calculate split indices based on class type
            n_samples = signals.shape[0]
            
            # Randomly shuffle indices for train/val/test split
            indices = np.random.permutation(n_samples)
            
            # For known classes: 80% train, 10% val, 10% test
            # For unknown classes: 0% train, 50% val, 50% test
            if not is_unk:
                n_train = int(n_samples * self.train_split)
                n_val = int(n_samples * self.val_split)
                
                train_indices = indices[:n_train]
                val_indices = indices[n_train:n_train + n_val]
                test_indices = indices[n_train + n_val:]
            else:
                n_val = int(n_samples * 0.5)
                
                train_indices = []
                val_indices = indices[:n_val]
                test_indices = indices[n_val:]
            
            # Select indices based on split type
            selected_indices = None
            if self.split == 'train':
                if not is_unk:
                    selected_indices = train_indices
            elif self.split == 'val':
                selected_indices = val_indices
            else:  # test
                selected_indices = test_indices
            
            # Add selected samples
            if selected_indices is not None and len(selected_indices) > 0:
                for idx in selected_indices:
                    # Extract noisy and clean signals
                    noisy = signals[idx, :2, :]  # Channels 0,1: noisy (Real, Imag)
                    clean = signals[idx, 2:, :]  # Channels 2,3: clean (Real, Imag)
                    
                    self.noisy_samples.append(noisy)
                    self.clean_samples.append(clean)
                    self.labels.append(label)
                    self.snrs.append(snr)
                    self.is_unknown.append(is_unk)
                    self.original_modulations.append(mod)
        
        # Convert to numpy arrays
        self.noisy_samples = np.array(self.noisy_samples, dtype=np.float32)
        self.clean_samples = np.array(self.clean_samples, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.snrs = np.array(self.snrs, dtype=np.float32)
        self.is_unknown = np.array(self.is_unknown, dtype=bool)
        self.original_modulations = np.array(self.original_modulations, dtype='U20')
        
        # Build fast per-class indices for known classes
        self.known_indices = np.where(self.labels >= 0)[0]
        self.indices_by_class = {k: np.where(self.labels == k)[0] for k in range(self.num_classes)}
        
        # Normalize if requested
        if self.normalize:
            # Normalize noisy samples
            noisy_energies = np.sqrt(np.mean(self.noisy_samples**2, axis=(1, 2)))
            mask = noisy_energies > 0
            self.noisy_samples[mask] = self.noisy_samples[mask] / (noisy_energies[mask, np.newaxis, np.newaxis] * 3.0)
            # Clean samples are already normalized relative to noisy by the same factor in data generation
            self.clean_samples[mask] = self.clean_samples[mask] / (noisy_energies[mask, np.newaxis, np.newaxis] * 3.0)
        
        logger.info(f"Loaded {len(self.noisy_samples)} denoising samples for {self.split} set")
        logger.info(f"Known classes: {len(self.noisy_samples[~self.is_unknown])} samples")
        logger.info(f"Unknown classes: {len(self.noisy_samples[self.is_unknown])} samples")
    
    def __len__(self):
        """Return number of samples"""
        return len(self.noisy_samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Returns:
            noisy_sample: Noisy I/Q signal data [2, 128] (possibly masked)
            clean_sample: Clean I/Q signal data [2, 128] (possibly masked)
            label: Class label (or -1 for unknown)
            info: Dictionary with SNR, is_unknown flag, and original modulation
        """
        noisy_sample = self.noisy_samples[idx].copy()
        clean_sample = self.clean_samples[idx].copy()
        label = self.labels[idx]
        
        # Apply random block masking if enabled (only to noisy input, NOT to clean target)
        if self.mask_ratio > 0.0:
            noisy_sample = apply_random_block_masking(noisy_sample, self.mask_ratio)
        
        # Convert to torch tensors
        noisy_sample = torch.from_numpy(noisy_sample).float()
        clean_sample = torch.from_numpy(clean_sample).float()
        
        # Prepare additional info
        info = {
            'is_unknown': self.is_unknown[idx],
            'snr': self.snrs[idx],
            'original_modulation': self.original_modulations[idx]
        }
        
        return noisy_sample, clean_sample, label, info
    
    def get_known_unknown_split(self):
        """Return indices of known and unknown samples for evaluation"""
        known_indices = np.where(~self.is_unknown)[0]
        unknown_indices = np.where(self.is_unknown)[0]
        return known_indices, unknown_indices
    
    def get_per_class_indices(self):
        """Return indices for each modulation class"""
        per_class_indices = {}
        unique_modulations = np.unique(self.original_modulations)
        for mod in unique_modulations:
            indices = np.where(self.original_modulations == mod)[0]
            per_class_indices[mod] = indices
        return per_class_indices


def get_rml_denoising_dataloaders(
    data_path: str,
    experiment_setting: int,
    batch_size: int = 64,
    num_workers: int = 4,
    snr_range: Tuple[int, int] = (-20, 20),
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    normalize: bool = False,
    seed: int = 42,
    mask_ratio: float = 0.0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for RML2016 denoising task
    
    Args:
        data_path: Path to denoising dataset (4-channel format)
        experiment_setting: Which experimental setting to use (1-12)
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        snr_range: SNR range to include
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        test_split: Proportion of data for testing
        normalize: Whether to normalize I/Q samples
        seed: Random seed
        mask_ratio: Ratio of signal length to mask (0.0-1.0)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = RML2016DenoisingDataset(
        data_path=data_path,
        experiment_setting=experiment_setting,
        split='train',
        snr_range=snr_range,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        normalize=normalize,
        seed=seed,
        mask_ratio=mask_ratio
    )
    
    val_dataset = RML2016DenoisingDataset(
        data_path=data_path,
        experiment_setting=experiment_setting,
        split='val',
        snr_range=snr_range,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        normalize=normalize,
        seed=seed,
        mask_ratio=mask_ratio
    )
    
    test_dataset = RML2016DenoisingDataset(
        data_path=data_path,
        experiment_setting=experiment_setting,
        split='test',
        snr_range=snr_range,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        normalize=normalize,
        seed=seed,
        mask_ratio=mask_ratio
    )
    
    # Create dataloaders
    train_loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    if num_workers and num_workers > 0:
        train_loader_kwargs['prefetch_factor'] = 4
        train_loader_kwargs['persistent_workers'] = True
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    
    val_loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    if num_workers and num_workers > 0:
        val_loader_kwargs['prefetch_factor'] = 4
        val_loader_kwargs['persistent_workers'] = True
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)
    
    test_loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    if num_workers and num_workers > 0:
        test_loader_kwargs['prefetch_factor'] = 4
        test_loader_kwargs['persistent_workers'] = True
    test_loader = DataLoader(test_dataset, **test_loader_kwargs)
    
    logger.info(f"Created denoising dataloaders - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches, Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

