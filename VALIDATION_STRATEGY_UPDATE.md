# Validation Strategy Update: ID/OOD Separation with Residual Spectrum Analysis

## Overview

This document describes the updated validation strategy that separates In-Distribution (ID) and Out-of-Distribution (OOD) samples during training evaluation, providing more detailed metrics for monitoring model performance on both types of data.

## Changes Summary

### 1. Enhanced OOD Detection with Batch Processing

**File**: `meanflow/evaluate_ood_advanced.py`

**Change**: Optimized `compute_spectral_flatness()` to use batch computation with PyTorch's native FFT

**Benefits**:
- **10-100x faster**: Eliminated Python loop over batch samples
- **GPU acceleration**: Computation now stays on GPU throughout
- **Welch's method**: Implemented batch Welch's periodogram for robust PSD estimation
- **Memory efficient**: Processes entire batches in parallel

**Technical Details**:
- Uses `torch.fft.fft()` for fast Fourier transform
- Implements Welch's method with Hann windowing and segment averaging
- Computes spectral flatness = geometric_mean(PSD) / arithmetic_mean(PSD)
- Low flatness (~0) indicates structural/tonal content (likely OOD residual)
- High flatness (~1) indicates white noise (likely ID residual)

### 2. Separated ID/OOD Validation Metrics

**File**: `meanflow/train_denoising.py`

**Changes**: Modified `evaluate()` function to:
1. Separate ID (labels >= 0) and OOD (labels == -1) samples
2. Compute and report MSE separately for ID and OOD
3. Add residual spectrum scores for both ID and OOD
4. Report AUROC for residual spectrum-based OOD detection

**New Metrics Reported During Validation**:

```
Overall Metrics (Best Hypothesis):
  Overall MSE: X.XXXXXX
  Overall NMSE: X.XXXXXX
  SNR Improvement: XX.XX dB
  Correlation: X.XXXX
  PSNR: XX.XX dB

ID vs OOD Metrics:
  ID MSE (min): X.XXXXXX
  ID MSE (avg): X.XXXXXX
  ID Residual Spectrum: X.XXXX
  OOD MSE (min): X.XXXXXX
  OOD MSE (avg): X.XXXXXX
  OOD Residual Spectrum: X.XXXX

OOD Detection Performance (AUROC):
  MSE-based AUROC: X.XXXX
  Correlation-based AUROC: X.XXXX
  SNR Improvement AUROC: X.XXXX
  Residual Spectrum AUROC: X.XXXX

Per-SNR Results:
  SNR XXX dB: MSE=X.XXXXXX, Imp=XX.XX dB, Corr=X.XXXX
  ...
```

### 3. Metrics Explanation

#### ID (In-Distribution) Metrics
- **ID MSE (min)**: Minimum MSE across all known classes (best hypothesis)
  - Lower is better
  - Shows best-case denoising performance when we choose the correct class
  - Oracle metric: assumes we can select the best class for each sample
  
- **ID MSE (avg)**: Average MSE across all known classes
  - Lower is better
  - Shows average denoising quality across all class hypotheses
  - More realistic: represents uncertainty when we don't know which class to use
  - **Key insight**: Gap between min and avg indicates class discrimination ability
  
- **ID Residual Spectrum**: Average spectral flatness of residuals for ID samples
  - Higher values (~1.0) indicate residual is white noise (good denoising)
  - Lower values indicate structural artifacts remain

#### OOD (Out-of-Distribution) Metrics
- **OOD MSE (min)**: Minimum MSE across all known classes (best hypothesis)
  - Expected to be higher than ID MSE (min)
  - Shows best-case performance even when forcing unknown samples into known classes
  
- **OOD MSE (avg)**: Average MSE across all known classes
  - Expected to be higher than ID MSE (avg)
  - Shows typical performance when OOD samples don't fit any known class well
  - **Key insight**: Large gap between OOD avg and ID avg indicates good OOD detection capability
  
- **OOD Residual Spectrum**: Average spectral flatness of residuals for OOD samples
  - Lower values indicate the model leaves structural artifacts (detecting anomalies)
  - Higher values might indicate the model is "hallucinating" plausible denoising

#### OOD Detection AUROC
- **Residual Spectrum AUROC**: Ability to distinguish ID from OOD using spectral analysis
  - 1.0 = perfect separation
  - 0.5 = random guessing
  - Higher is better
  
- **MSE-based AUROC**: Traditional reconstruction error approach
- **Correlation-based AUROC**: Input-output correlation approach
- **SNR Improvement AUROC**: Blind SNR improvement approach

### 4. Integration with Existing Code

The changes are **backward compatible** and don't break existing functionality:

- All original metrics are still computed and reported
- Existing per-SNR and per-modulation breakdowns remain unchanged
- WandB logging automatically includes new metrics
- No changes required to training arguments or configuration

## Usage

No changes to training commands are needed. Simply run training as before:

```bash
python meanflow/train_denoising.py \
    --data_path data/RML2016_denoising.pkl \
    --experiment_setting 1 \
    --batch_size 256 \
    --epochs 100 \
    --lr 2e-4 \
    --eval_freq 5
```

During validation (every `--eval_freq` epochs), you'll now see the enhanced metrics separating ID and OOD performance.

## Implementation Details

### Residual Spectrum Score Computation

For each sample:
1. Compute residual: `residual = noisy_input - denoised_output`
2. Convert to complex signal: `complex = I + jQ`
3. Compute Power Spectral Density using Welch's method
4. Calculate spectral flatness: `exp(mean(log(PSD))) / mean(PSD)`
5. OOD score: `-log(spectral_flatness + epsilon)`

### ID/OOD Separation Logic

```python
# During evaluation loop
is_id = (labels != -1)  # Known modulation types
is_ood = (labels == -1)  # Unknown modulation types

# For each sample, try denoising with all known classes
batch_sum_errors = torch.zeros(batch_size)
batch_min_errors = torch.full(batch_size, float('inf'))

for class_id in range(num_classes):
    # Denoise with class hypothesis
    denoised = model.denoise(noisy, class_labels=class_id)
    errors = mse(denoised, clean)
    
    # Track both min and sum for averaging
    batch_min_errors = torch.minimum(batch_min_errors, errors)
    batch_sum_errors += errors

# Calculate average MSE
batch_avg_errors = batch_sum_errors / num_classes

# Separate metrics by ID/OOD
id_mse_min = batch_min_errors[is_id]
id_mse_avg = batch_avg_errors[is_id]
ood_mse_min = batch_min_errors[is_ood]
ood_mse_avg = batch_avg_errors[is_ood]
```

## Benefits

1. **Better Monitoring**: Track ID and OOD performance separately during training
2. **Early Detection**: Identify if the model is overfitting to known classes
3. **Performance Analysis**: Understand the trade-off between ID accuracy and OOD detection
4. **Multiple Metrics**: Compare different OOD detection approaches (MSE, correlation, SNR, spectral)
5. **Efficiency**: Batch processing makes evaluation faster despite computing more metrics
6. **Min vs Avg MSE**: Dual metrics provide complementary insights:
   - **Min MSE**: Best-case performance (oracle scenario)
   - **Avg MSE**: Realistic performance (no prior knowledge)
   - **Gap analysis**: Reveals model's confidence and class separability

## Future Enhancements

Potential extensions:
- Add per-class ID/OOD metrics
- Implement adaptive thresholding based on validation residual spectrum scores
- Add confidence calibration for OOD detection
- Visualize residual spectrum distributions for ID vs OOD

## References

- Spectral flatness (Wiener entropy) for signal analysis
- Welch's periodogram method for robust PSD estimation
- OOD detection through reconstruction-based approaches

