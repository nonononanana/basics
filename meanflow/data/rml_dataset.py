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


class RML2016Dataset(Dataset):
    """
    Dataset class for RML2016.10a modulation classification
    Supports open-set recognition with known/unknown class splits
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
        return_snr: bool = False,  # Return SNR as additional info
        seed: int = 42,
        precompute_negatives: bool = True  # Pre-compute negative samples
    ):
        """
        Initialize RML2016 Dataset
        
        Args:
            data_path: Path to RML2016.10a_dict.pkl file
            experiment_setting: Which experimental setting (1-12) to use
            split: Dataset split - 'train', 'val', or 'test'
            snr_range: Range of SNR values to include
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
            test_split: Proportion of data for testing
            normalize: Whether to normalize I/Q samples
            return_snr: Whether to return SNR information
            seed: Random seed for reproducibility
            precompute_negatives: Whether to pre-compute negative samples (faster training, more memory)
        """
        super().__init__()
        
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}. Must be 'train', 'val', or 'test'"
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
        
        self.data_path = data_path
        self.experiment_setting = experiment_setting
        self.split = split
        self.snr_range = snr_range
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.normalize = normalize
        self.return_snr = return_snr
        self.seed = seed
        self.precompute_negatives = precompute_negatives
        
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
        """Load and prepare RML2016.10a dataset with train/val/test splits"""
        # Load the pickle file
        with open(self.data_path, 'rb') as f:
            data_dict = pickle.load(f, encoding='latin1')
        
        # Separate data by modulation type and SNR
        self.samples = []
        self.labels = []
        self.snrs = []
        self.is_unknown = []  # Track whether sample is from unknown class
        self.original_modulations = []  # Track original modulation type for all samples
        
        np.random.seed(self.seed)
        
        # Process each (modulation, SNR) key
        for (mod, snr), signals in data_dict.items():
            # Filter by SNR range
            if snr < self.snr_range[0] or snr > self.snr_range[1]:
                continue
                
            # Determine if this is known or unknown class
            if mod in self.known_classes:
                label = self.class_to_idx[mod]
                is_unk = False
            elif mod in self.unknown_classes:
                label = -1  # Unknown class label
                is_unk = True
            else:
                logger.info(f"Error:  {mod} is outer data from current dataset")
                continue  # Skip if not in either list
            
            # Calculate split indices based on class type
            n_samples = signals.shape[0]  # Typically 1000 samples per (mod, SNR) pair
            
            # Randomly shuffle indices for train/val/test split
            indices = np.random.permutation(n_samples)
            
            # For known classes: 80% train, 10% val, 10% test
            # For unknown classes: 0% train, 50% val, 50% test (to balance with known in val/test)
            if not is_unk:
                # Known class split
                n_train = int(n_samples * self.train_split)  # 80% -> 800
                n_val = int(n_samples * self.val_split)  # 10% -> 100
                # Remaining samples go to test set (10% -> 100)
                
                train_indices = indices[:n_train]
                val_indices = indices[n_train:n_train + n_val]
                test_indices = indices[n_train + n_val:]
            else:
                # Unknown class split: only for val/test (no training)
                # Split 50/50 between val and test for better balance
                n_val = int(n_samples * 0.5)  # 50% -> 500
                # Remaining 50% goes to test
                
                train_indices = []  # No training data for unknown
                val_indices = indices[:n_val]
                test_indices = indices[n_val:]
            
            # Select indices based on split type
            selected_indices = None
            if self.split == 'train':
                # Training set: only use known classes (exclude unknown)
                if not is_unk:
                    selected_indices = train_indices
            elif self.split == 'val':
                # Validation set: use both known (10%) and unknown (50%)
                selected_indices = val_indices
            else:  # test
                # Test set: use both known (10%) and unknown (50%)
                selected_indices = test_indices
            
            # Add selected samples in batch for efficiency
            if selected_indices is not None and len(selected_indices) > 0:
                batch_samples = [signals[idx] for idx in selected_indices]
                batch_labels = [label] * len(selected_indices)
                batch_snrs = [snr] * len(selected_indices)
                batch_is_unknown = [is_unk] * len(selected_indices)
                batch_modulations = [mod] * len(selected_indices)  # Track original modulation
                
                # Extend lists with batched data (more efficient than individual appends)
                self.samples.extend(batch_samples)
                self.labels.extend(batch_labels)
                self.snrs.extend(batch_snrs)
                self.is_unknown.extend(batch_is_unknown)
                self.original_modulations.extend(batch_modulations)
        
        # Convert to numpy arrays
        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.snrs = np.array(self.snrs, dtype=np.float32)
        self.is_unknown = np.array(self.is_unknown, dtype=bool)
        self.original_modulations = np.array(self.original_modulations, dtype='U20')  # Unicode string array
        
        # Build fast per-class indices for known classes (used for negative mixing)
        # Known classes are labeled 0..num_classes-1; unknowns use -1
        self.known_indices = np.where(self.labels >= 0)[0]
        self.indices_by_class = {k: np.where(self.labels == k)[0] for k in range(self.num_classes)}
        # Precompute other-class choices for each class for O(1) sampling
        self.other_label_choices = {
            k: np.array([j for j in range(self.num_classes) if j != k], dtype=np.int64)
            for k in range(self.num_classes)
        }

        # Normalize if requested
        if self.normalize:
            # Normalize each sample independently to [-1, 1] using vectorized operations
            # Compute energy for all samples at once
            energies = np.sqrt(np.mean(self.samples**2, axis=(1, 2)))
            # Avoid division by zero
            mask = energies > 0
            self.samples[mask] = self.samples[mask] / (energies[mask, np.newaxis, np.newaxis] * 3.0)
        
        logger.info(f"Loaded {len(self.samples)} samples for {self.split} set")
        logger.info(f"Known classes: {len(self.samples[~self.is_unknown])} samples")
        logger.info(f"Unknown classes: {len(self.samples[self.is_unknown])} samples")
        
        # Pre-compute negative samples if requested and in training mode
        if self.precompute_negatives and (self.split == 'train' or self.split == 'val'): 
            self._precompute_negative_samples()
    
    def _precompute_negative_samples(self):
        """
        Pre-compute negative samples for all training data.
        """
        start_time = time.time()
        n_samples = len(self.samples)
        
        logger.info("Pre-computing negative samples...")
        
        # Initialize storage for negative samples
        self.negative_samples = []
        
        # Process each sample
        for idx in range(n_samples):
            # Generate one negative sample for this positive sample using mixing strategy
            neg_sample = self._create_mixed_negative_sample(idx)
            self.negative_samples.append(neg_sample)
            
            # Log progress every 5000 samples
            if (idx + 1) % 5000 == 0 or idx == n_samples - 1:
                progress = (idx + 1) / n_samples * 100
                logger.info(f"  Progress: {progress:.1f}% ({idx + 1}/{n_samples} samples)")
        
        # Convert to single numpy array for more efficient storage
        self.negative_samples = np.array(self.negative_samples, dtype=np.float32)
        
        # Calculate memory usage
        memory_mb = self.negative_samples.nbytes / (1024 * 1024)
        elapsed_time = time.time() - start_time
        
        logger.info(f"Pre-computation completed in {elapsed_time:.2f} seconds")
        logger.info(f"Negative samples memory usage: {memory_mb:.2f} MB")
        logger.info(f"Shape: {self.negative_samples.shape} (samples x channels x time)")
        
    def __len__(self):
        """Return number of samples"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Returns:
            positive_sample: Original I/Q signal data [2, 128]
            negative_sample: Corrupted I/Q signal data [2, 128]
            label: Class label (or -1 for unknown)
            Additional info dict with SNR and is_unknown flag
        """
        # Get the original sample (positive)
        positive_sample = self.samples[idx].copy()
        label = self.labels[idx]
        
        # Get negative sample (for training and validation)
        negative_sample = None
        if self.split in ['train', 'val']:
            if self.precompute_negatives:
                # Use pre-computed negative sample
                negative_sample = self.negative_samples[idx].copy()
            else:
                # Generate mixed negative sample on-the-fly (70% current corrupted + 30% other class)
                negative_sample = self._create_mixed_negative_sample(idx)
        else:
            # For test split, create a dummy negative sample (not used in evaluation but needed for collation)
            negative_sample = np.zeros_like(positive_sample)
        
        # Convert to torch tensors
        positive_sample = torch.from_numpy(positive_sample).float()
        negative_sample = torch.from_numpy(negative_sample).float()
        
        # Prepare additional info
        info = {
            'is_unknown': self.is_unknown[idx],
            'snr': self.snrs[idx] if self.return_snr else None,
            'original_modulation': self.original_modulations[idx]  # Add original modulation
        }
        
        return positive_sample, negative_sample, label, info
    
    def _create_mixed_negative_sample(self, idx: int):
        """
        70/30 strategy:
          - 70%: create_negative_sample from current-class signal
          - 30%: create_negative_sample from an other-class signal (OOD)
        """
        sample = self.samples[idx].copy()
        label = int(self.labels[idx])
        if np.random.rand() < 0.7:
            return self._create_negative_sample(sample).astype(np.float32)
        else:
            other = self._sample_other_class_signal(label)
            return self._create_negative_sample(other).astype(np.float32)

    def _sample_other_class_signal(self, current_label: int):
        """
        Sample a signal from a different known class than current_label.
        Falls back to any known sample if class buckets are empty (shouldn't happen).
        """
        # If current label unknown or single-class edge case, fallback to any known sample
        if current_label < 0 or self.num_classes <= 1 or len(self.known_indices) == 0:
            fallback_idx = int(np.random.choice(self.known_indices)) if len(self.known_indices) > 0 else 0
            return self.samples[fallback_idx].copy()
        # Choose a different class
        choices = self.other_label_choices[current_label]
        other_label = int(np.random.choice(choices)) if len(choices) > 0 else (current_label + 1) % self.num_classes
        pool = self.indices_by_class.get(other_label, None)
        if pool is None or len(pool) == 0:
            # Fallback to any known sample
            fallback_idx = int(np.random.choice(self.known_indices)) if len(self.known_indices) > 0 else 0
            return self.samples[fallback_idx].copy()
        other_idx = int(np.random.choice(pool))
        return self.samples[other_idx].copy()

    def _create_negative_sample(self, sample):
        """
        Create a negative sample by applying heavy corruptions to deviate 
        from the original modulation pattern
        
        Args:
            sample: numpy array of shape [2, 128]
        
        Returns:
            Heavily corrupted sample
        """
        # Apply multiple random corruptions to create negative samples
        
        # Track whether any corruption has been applied; ensure at least one
        any_corruption_applied = False
        
        # 1. Add large Gaussian noise (optional)
        if np.random.rand() > 0.8:
            noise_power = np.random.uniform(0.1, 0.5)
            noise = np.random.randn(*sample.shape) * np.sqrt(noise_power)
            sample = sample + noise
            any_corruption_applied = True
        
        # 2. Carrier Frequency Offset (CFO)
        if np.random.rand() > 0.7:
            # CFO causes phase rotation that increases over time
            cfo_freq = np.random.uniform(-0.1, 0.1)  # Normalized frequency offset
            time_indices = np.arange(sample.shape[1])
            phase_shift = 2 * np.pi * cfo_freq * time_indices
            cos_shift = np.cos(phase_shift)
            sin_shift = np.sin(phase_shift)
            i_channel = sample[0] * cos_shift - sample[1] * sin_shift
            q_channel = sample[0] * sin_shift + sample[1] * cos_shift
            sample = np.stack([i_channel, q_channel], axis=0)
            if np.abs(cfo_freq) >= 0.01:
                any_corruption_applied = True
        
        # 3. Phase noise
        if np.random.rand() > 0.7:
            # Random walk phase noise
            phase_noise_std = np.random.uniform(0.05, 0.2)
            phase_noise = np.cumsum(np.random.randn(sample.shape[1]) * phase_noise_std)
            cos_noise = np.cos(phase_noise)
            sin_noise = np.sin(phase_noise)
            i_channel = sample[0] * cos_noise - sample[1] * sin_noise
            q_channel = sample[0] * sin_noise + sample[1] * cos_noise
            sample = np.stack([i_channel, q_channel], axis=0)
            if phase_noise_std >= 0.12:
                any_corruption_applied = True
        
        # 4. I/Q imbalance and DC offset
        if np.random.rand() > 0.7:
            # I/Q imbalance
            gain_imbalance = np.random.uniform(0.5, 1.5)
            sample[0] *= gain_imbalance
            
            # DC offset
            dc_offset_i = np.random.uniform(-0.3, 0.3)
            dc_offset_q = np.random.uniform(-0.3, 0.3)
            sample[0] += dc_offset_i
            sample[1] += dc_offset_q
            if abs(gain_imbalance - 1.0) >= 0.3 or np.sqrt(dc_offset_i**2 + dc_offset_q**2) >= 0.25:
                any_corruption_applied = True
        
        # 5. Sampling rate shift (resample)
        if np.random.rand() > 0.7:
            # Simple resampling by interpolation
            resample_factor = np.random.uniform(0.8, 1.2)
            old_indices = np.arange(sample.shape[1])
            new_length = int(sample.shape[1] * resample_factor)
            new_indices = np.linspace(0, sample.shape[1] - 1, new_length)
            
            # Interpolate both I and Q channels
            f_i = interpolate.interp1d(old_indices, sample[0], kind='linear', fill_value='extrapolate')
            f_q = interpolate.interp1d(old_indices, sample[1], kind='linear', fill_value='extrapolate')
            
            # Resample and truncate/pad to original length
            resampled_i = f_i(new_indices)
            resampled_q = f_q(new_indices)
            
            if new_length > sample.shape[1]:
                sample[0] = resampled_i[:sample.shape[1]]
                sample[1] = resampled_q[:sample.shape[1]]
            else:
                sample[0, :new_length] = resampled_i
                sample[1, :new_length] = resampled_q
                # Pad with noise
                sample[0, new_length:] = np.random.randn(sample.shape[1] - new_length) * 0.1
                sample[1, new_length:] = np.random.randn(sample.shape[1] - new_length) * 0.1
            if np.abs(resample_factor-1) >= 0.05:
                any_corruption_applied = True
        
        # 6. Non-linear distortion
        if np.random.rand() > 0.7:
            # Apply polynomial non-linearity
            alpha = np.random.uniform(0.1, 0.3)
            magnitude = np.sqrt(sample[0]**2 + sample[1]**2)
            distortion = 1 + alpha * magnitude**2
            sample = sample * distortion
        
        # 7. Random phase/amplitude jumps
        if np.random.rand() > 0.7:
            # Create random discontinuities
            num_jumps = np.random.randint(1, 4)
            jump_positions = np.random.choice(sample.shape[1], num_jumps, replace=False)
            phase_jump = 0
            amp_jump = 0
            for pos in jump_positions:
                if np.random.rand() > 0.5:
                    # Phase jump
                    phase_jump = np.random.uniform(-np.pi, np.pi)
                    cos_jump = np.cos(phase_jump)
                    sin_jump = np.sin(phase_jump)
                    i_temp = sample[0, pos:] * cos_jump - sample[1, pos:] * sin_jump
                    q_temp = sample[0, pos:] * sin_jump + sample[1, pos:] * cos_jump
                    sample[0, pos:] = i_temp
                    sample[1, pos:] = q_temp
                else:
                    # Amplitude jump
                    amp_jump = np.random.uniform(0.3, 2.0)
                    sample[:, pos:] *= amp_jump
            if 0.7 <= amp_jump <= 1.3 or np.abs(phase_jump) >= np.pi / 3:
                any_corruption_applied = True
        
        # 8. Random signal substitution
        if np.random.rand() > 0.7:
            # Replace random segments with noise
            num_segments = np.random.randint(1, 3)
            segment_length = np.random.randint(5, 20)
            replaced_length = 0
            for _ in range(num_segments):
                start_pos = np.random.randint(0, max(1, sample.shape[1] - segment_length))
                end_pos = min(start_pos + segment_length, sample.shape[1])
                sample[:, start_pos:end_pos] = np.random.randn(2, end_pos - start_pos) * 0.5
                replaced_length += end_pos - start_pos
                assert end_pos - start_pos == segment_length
            if replaced_length >= 16:
                any_corruption_applied = True
        
        # 9. Apply random filtering
        if np.random.rand() > 0.7:
            # Design a random filter (lowpass or highpass)
            filter_type = np.random.choice(['lowpass', 'highpass']) 
            if filter_type == 'lowpass':
                # Lowpass filter
                cutoff = np.random.uniform(0.1, 0.4)
                b, a = signal.butter(3, cutoff, 'low')
            else:
                # Highpass filter
                cutoff = np.random.uniform(0.1, 0.3)
                b, a = signal.butter(3, cutoff, 'high')
            
            sample[0] = signal.filtfilt(b, a, sample[0])
            sample[1] = signal.filtfilt(b, a, sample[1])
            if filter_type == 'lowpass':
                if cutoff <= 0.2:
                    any_corruption_applied = True
            else:
                if cutoff >= 0.2:
                    any_corruption_applied = True
        
        # Ensure at least one corruption is applied
        if not any_corruption_applied:
            noise_power = np.random.uniform(0.1, 0.5)
            noise = np.random.randn(*sample.shape) * np.sqrt(noise_power)
            sample = sample + noise
        
        return sample
    
    def get_known_unknown_split(self):
        """
        Return indices of known and unknown samples for evaluation
        """
        known_indices = np.where(~self.is_unknown)[0]
        unknown_indices = np.where(self.is_unknown)[0]
        return known_indices, unknown_indices
    
    def get_per_class_indices(self):
        """
        Return indices for each modulation class (both known and unknown)
        """
        per_class_indices = {}
        unique_modulations = np.unique(self.original_modulations)
        for mod in unique_modulations:
            indices = np.where(self.original_modulations == mod)[0]
            per_class_indices[mod] = indices
        return per_class_indices


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
        seed: int = 42
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
        """
        super().__init__()
        
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}. Must be 'train', 'val', or 'test'"
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
        
        self.data_path = data_path
        self.experiment_setting = experiment_setting
        self.split = split
        self.snr_range = snr_range
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.normalize = normalize
        self.seed = seed
        
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
            noisy_sample: Noisy I/Q signal data [2, 128]
            clean_sample: Clean I/Q signal data [2, 128]
            label: Class label (or -1 for unknown)
            info: Dictionary with SNR, is_unknown flag, and original modulation
        """
        noisy_sample = self.noisy_samples[idx].copy()
        clean_sample = self.clean_samples[idx].copy()
        label = self.labels[idx]
        
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
    seed: int = 42
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
        seed=seed
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
        seed=seed
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
        seed=seed
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


def get_rml_dataloaders(
    data_path: str,
    experiment_setting: int,
    batch_size: int = 64,
    num_workers: int = 16,
    snr_range: Tuple[int, int] = (-20, 20),
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    normalize: bool = False,
    return_snr: bool = True,
    seed: int = 42,
    precompute_negatives: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for RML2016.10a dataset
    
    Args:
        data_path: Path to RML2016.10a_dict.pkl
        experiment_setting: Which experimental setting to use (1-12)
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        snr_range: SNR range to include
        train_split: Proportion of data for training (default: 0.8)
        val_split: Proportion of data for validation (default: 0.1)
        test_split: Proportion of data for testing (default: 0.1)
        normalize: Whether to normalize I/Q samples
        return_snr: Whether to return SNR information (default: True)
        seed: Random seed
        precompute_negatives: Whether to pre-compute negative samples (faster training, more memory)
    
    Returns:
        train_loader, val_loader, test_loader
    
    Note:
        - Training set: Contains only known classes (80% of each known class) with negative samples
        - Validation set: Contains known (10% of each known class) and unknown (50% of each unknown class)
        - Test set: Contains known (10% of each known class) and unknown (50% of each unknown class)
        - Unknown classes split 50/50 between val/test for better balance (not used in training)
    """
    # Create datasets
    train_dataset = RML2016Dataset(
        data_path=data_path,
        experiment_setting=experiment_setting,
        split='train',
        snr_range=snr_range,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        normalize=normalize,
        return_snr=return_snr,
        seed=seed,
        precompute_negatives=precompute_negatives
    )
    
    val_dataset = RML2016Dataset(
        data_path=data_path,
        experiment_setting=experiment_setting,
        split='val',
        snr_range=snr_range,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        normalize=normalize,
        return_snr=return_snr,
        seed=seed,
        precompute_negatives=precompute_negatives  # Now includes negative samples for validation
    )
    
    test_dataset = RML2016Dataset(
        data_path=data_path,
        experiment_setting=experiment_setting,
        split='test',
        snr_range=snr_range,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        normalize=normalize,
        return_snr=return_snr,
        seed=seed,
        precompute_negatives=False  # No negative samples needed for testing
    )
    
    # Create dataloaders
    # Build DataLoader kwargs with optional prefetch and persistent workers
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
    
    logger.info(f"Created dataloaders - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches, Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader
