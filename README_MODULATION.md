# Mean Flow for Open-Set Modulation Classification

## Overview

This project adapts the Mean Flow generative model for open-set automatic modulation classification on the RML2016.10a dataset. The system can:
- Classify known modulation types with high accuracy
- Detect and reject unknown modulation types using energy-based scoring
- Generate synthetic modulation signals conditioned on class labels
- Achieve strong class separation using ArcFace loss

## Key Features

### 1. Open-Set Recognition
- **Energy-Based OOD Detection**: Uses reconstruction energy to identify unknown modulations
- **Adaptive Thresholding**: Automatically tunes energy threshold for optimal F1 score
- **AUROC/AUPR Metrics**: Comprehensive evaluation of OOD detection performance

### 2. Conditional Generation
- **Class-Conditioned Diffusion**: Generate specific modulation types on demand
- **Mean Flow Framework**: Efficient single-step generation with deterministic mapping
- **EMA Stabilization**: Multiple EMA networks for stable generation

### 3. Advanced Training Techniques
- **ArcFace Loss**: Enforces angular margin between classes for better separability
- **Adaptive Weight Normalization**: Stabilizes training with adaptive loss weighting
- **Mixed Precision Training**: Faster training with automatic mixed precision
- **Regularization**: Weight decay and dropout to prevent overfitting

## Architecture

### Model Components

1. **ModulationUNet**: Custom UNet architecture for I/Q signal processing
   - 1D convolutions for temporal signal processing
   - Multi-scale feature extraction with skip connections
   - Self-attention modules for long-range dependencies
   - Class embedding for conditional generation

2. **MeanFlowModulation**: Main model class
   - Implements mean flow dynamics for generation
   - Energy scoring for OOD detection
   - ArcFace loss computation
   - EMA network management

3. **RML2016Dataset**: Data loader with open-set splits
   - 12 experimental settings with different known/unknown splits
   - Data augmentation (phase rotation, amplitude scaling, noise, time shift)
   - SNR-aware sampling
   - Proper train/test separation

## Dataset Configuration

The RML2016.10a dataset contains 11 modulation types:
- **Digital**: 8PSK, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK
- **Analog**: AM-DSB, AM-SSB, WBFM

### Experimental Settings

| Setting | Known Classes | Unknown Classes | Description |
|---------|--------------|-----------------|-------------|
| 1 | 9 | 2 | High known ratio (10a) |
| 2 | 6 | 5 | Balanced split (10a) |
| 3 | 4 | 7 | High unknown ratio (10a) |
| 4 | 3 | 8 | Extreme unknown ratio (10a) |
| 5-8 | Various | Various | Different splits (10b) |
| 9-12 | Various | Various | Different splits (04c) |

## Installation

```bash
# Clone the repository
git clone <repository_url>
cd py-meanflow

# Create conda environment
conda env create -f environment.yml
conda activate meanflow

# Install additional dependencies
pip install wandb scikit-learn
```

## Training

### Basic Training

```bash
python meanflow/train_modulation.py \
    --data_path data/RML2016.10a_dict.pkl \
    --experiment_setting 1 \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4
```

### Advanced Training with All Features

```bash
python meanflow/train_modulation.py \
    --data_path data/RML2016.10a_dict.pkl \
    --experiment_setting 1 \
    --epochs 200 \
    --batch_size 128 \
    --lr 2e-4 \
    --model_channels 64 \
    --num_blocks 2 \
    --dropout 0.15 \
    --weight_decay 1e-4 \
    --use_arcface \
    --arcface_margin 0.5 \
    --arcface_scale 30.0 \
    --mixed_precision \
    --warmup_epochs 10 \
    --grad_clip 1.0 \
    --eval_freq 5 \
    --wandb_project meanflow-modulation \
    --wandb_name exp1_arcface
```

### Key Training Parameters

- `--experiment_setting`: Choose from 1-12 for different known/unknown splits
- `--snr_min/snr_max`: Filter data by SNR range (default: -20 to 20 dB)
- `--model_channels`: Base channel dimension (reduce for smaller model)
- `--dropout`: Dropout rate for regularization (increase if overfitting)
- `--weight_decay`: L2 regularization strength
- `--use_arcface`: Enable ArcFace loss for better class separation
- `--energy_temperature`: Temperature for energy scoring (affects OOD detection)

## Evaluation Metrics

### Closed-Set Metrics
- **Accuracy**: Classification accuracy on known classes only
- **Per-Class Accuracy**: Individual accuracy for each modulation type
- **Confusion Matrix**: Detailed classification errors

### Open-Set Metrics
- **AUROC**: Area Under ROC Curve for OOD detection
- **AUPR**: Area Under Precision-Recall Curve
- **Open-Set Accuracy**: Combined accuracy for known classification and unknown rejection
- **TPR/FPR**: True/False positive rates for unknown detection

### Energy Metrics
- **Mean Energy (Known)**: Average energy for in-distribution samples
- **Mean Energy (Unknown)**: Average energy for out-of-distribution samples
- **Energy Threshold**: Optimal threshold for OOD detection

## Design Decisions

### 1. Energy Scoring
The energy function computes the negative log-sum-exp of reconstruction errors across all classes:
```python
energy = -T * log(sum(exp(-error_c/T)))
```
This provides a calibrated score where:
- Low energy → Known class (good reconstruction)
- High energy → Unknown class (poor reconstruction)

### 2. ArcFace Loss
Adds angular margin between classes in embedding space:
```python
cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
```
Benefits:
- Increases inter-class variance
- Decreases intra-class variance
- Improves OOD detection by creating clear decision boundaries

### 3. Adaptive Weight Normalization
Stabilizes training by normalizing gradients:
```python
weight = (loss.detach() + ε)^p
normalized_loss = loss / weight
```
This prevents gradient explosion and ensures stable convergence.

### 4. Data Augmentation
Applied during training to improve robustness:
- **Phase Rotation**: Random rotation in I/Q plane
- **Amplitude Scaling**: Scale signal power by 0.8-1.2×
- **Additive Noise**: Small Gaussian noise (SNR > 20dB)
- **Time Shift**: Circular shift by ±10 samples

### 5. Model Capacity Adjustment
To prevent overfitting on small datasets:
- Reduced `model_channels` from 128 to 64
- Added `dropout` (0.1-0.2) in UNet blocks
- Applied `weight_decay` (1e-4) for L2 regularization
- Used `class_dropout` for classifier-free guidance

## Experiment Tracking

The project uses Weights & Biases (wandb) for experiment tracking:

```python
# View training progress
wandb.ai/<entity>/<project>

# Logged metrics:
- Training: loss, reconstruction_loss, arcface_loss, learning_rate
- Evaluation: auroc, aupr, accuracy, energy_threshold
- Per-class: individual class accuracies
- System: GPU utilization, memory usage
```

## Results Interpretation

### Good Performance Indicators
- AUROC > 0.9 for OOD detection
- Clear energy separation (2-3× difference between known/unknown)
- Balanced per-class accuracies
- Low false positive rate (< 10%)

### Troubleshooting

**High Training Loss**
- Reduce learning rate
- Increase warmup epochs
- Check data normalization

**Poor OOD Detection**
- Enable ArcFace loss
- Increase model capacity
- Adjust energy temperature
- Add more augmentation

**Overfitting**
- Increase dropout (0.2-0.3)
- Increase weight decay (5e-4)
- Reduce model channels
- Add more data augmentation

## File Structure

```
meanflow/
├── data/
│   └── rml_dataset.py          # RML2016 dataset loader
├── models/
│   ├── meanflow_modulation.py  # Mean flow model for modulation
│   └── unet_modulation.py      # UNet architecture for signals
├── train_modulation.py          # Main training script
└── scripts/
    └── run_experiments.sh       # Batch experiment runner
```

## Citation

If you use this code, please cite:
```bibtex
@article{meanflow2024,
  title={Mean Flow for Open-Set Modulation Classification},
  author={...},
  year={2024}
}
```

## License

This project is licensed under the CC-BY-NC license.
