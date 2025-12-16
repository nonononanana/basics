# OOD Evaluation Logic Review - Blind/Realistic Methods

## Summary of Changes

All OOD scoring methods have been updated to be **truly blind** - they only use the noisy input signal and do NOT use:
- ❌ Clean/ground truth signals
- ❌ True class labels (during testing)

This makes the evaluation realistic for real-world OOD detection scenarios.

---

## Fixed Methods (Previously Used Clean Signal)

### 1. `compute_ood_score_min_error`
**Before**: Compared denoised output to clean signal ❌  
**After**: Uses estimated SNR of denoised outputs ✅

**Logic**: 
- Try denoising with all known class labels
- Estimate SNR of each denoised output (blind estimator)
- Return `-max(SNR)` as OOD score
- **Intuition**: ID samples produce high-SNR outputs → low OOD score; OOD samples produce low-SNR outputs → high OOD score

### 2. `compute_ood_score_avg_error`
**Before**: Compared averaged denoised to clean signal ❌  
**After**: Uses average estimated SNR across all classes ✅

**Logic**:
- Try denoising with all known class labels
- Compute average SNR across all outputs
- Return `-avg(SNR)` as OOD score
- **Intuition**: ID samples have high average SNR; OOD samples have low average SNR

### 3. `compute_ood_score_improvement`
**Before**: Compared error before/after using clean signal ❌  
**After**: Measures output consistency across classes ✅

**Logic**:
- Try denoising with all known class labels
- Compute variance of outputs across different classes
- Return variance as OOD score
- **Intuition**: ID samples produce consistent outputs across classes; OOD samples produce inconsistent outputs (model is confused)

---

## Already Blind Methods (No Changes Needed)

### 4. `compute_ood_score_snr_improvement`
✅ **Already blind** - uses blind SNR estimator

**Logic**:
- Estimate SNR before denoising (input)
- Estimate SNR after denoising (for all classes)
- Return `-max(SNR_improvement)` as OOD score
- **Intuition**: ID samples show SNR improvement; OOD samples don't improve or get worse

### 5. `compute_ood_score_correlation`
✅ **Already blind** - only uses input and output

**Logic**:
- Compute correlation between input and denoised output
- Try all classes, keep maximum correlation
- Return `-max(correlation)` as OOD score
- **Intuition**: ID samples have high input-output correlation; OOD samples have low correlation

### 6. `compute_ood_score_mdrc` (NEW - Multi-Domain Residual Complexity)
✅ **Fully blind** - uses only noisy input

**Logic**:
- **Statistics Phase** (on ID validation data with known labels):
  - Denoise ID samples using their TRUE labels
  - Extract 3D features from residuals: [amplitude, phase, spectral]
  - Compute mean μ and covariance Σ of ID residual features
  
- **Testing Phase** (blind - no labels):
  - For each test sample, try ALL class labels
  - Extract 3D features from each residual
  - Compute Mahalanobis distance: `sqrt((v - μ)^T Σ^-1 (v - μ))`
  - Return minimum distance across all classes
  
- **Intuition**: 
  - ID samples: At least one class (the true one) produces "normal" residuals → low distance
  - OOD samples: ALL classes produce abnormal residuals → high distance

---

## Feature Extraction for MDRC

### Three-Domain Features from Residual `r[n] = input[n] - denoised[n]`

1. **Amplitude Domain** (for QAM, AM, PAM):
   - `v_amp = Kurtosis(|r[n]|)`
   - Measures "peakiness" of residual amplitude

2. **Phase/Frequency Domain** (for GFSK, CPFSK, PSK):
   - `φ[n] = unwrap(angle(r[n]))`
   - `d[n] = φ[n] - 2φ[n-1] + φ[n-2]` (2nd order difference)
   - `v_phase = Kurtosis(d[n])`
   - Measures smoothness of instantaneous frequency

3. **Spectral Domain** (for SSB, DSB, WBFM):
   - `PSD = |FFT(r[n])|^2`
   - `SFM = geometric_mean(PSD) / arithmetic_mean(PSD)`
   - `v_spec = -log10(SFM)`
   - Measures spectral flatness (high SFM = white noise = good denoising)

---

## Critical Logic Points

### ✅ During Statistics Computation (ID Validation Data)
```python
# For ID samples, we DO have true labels
labels = labels[known_mask].to(device)
denoised = model.denoise(x_noisy=noisy_samples, class_labels=labels, num_steps=1)
features = extract_residual_features(noisy_samples, denoised)
```
**This is correct**: We use true labels to establish what "normal" residuals look like.

### ✅ During OOD Testing (Test Data)
```python
# Try ALL classes blindly (no true labels used)
for class_id in range(num_classes):
    class_labels = torch.full((batch_size,), class_id, dtype=torch.long, device=device)
    denoised = model.denoise(x_noisy=noisy_signal, class_labels=class_labels, num_steps=1)
    # Compute OOD score...
```
**This is correct**: We don't know true labels, so we try all possibilities.

### ✅ Labels Only for Metrics
```python
# Labels ONLY used after scoring, for evaluation metrics
is_unknown = (labels == -1).long()  # 0=ID, 1=OOD
auroc = roc_auc_score(all_labels, all_ood_scores)
```
**This is correct**: Labels are only used to compute AUROC/AUPR, not for OOD scoring.

---

## Usage Example

```bash
# Evaluate with MDRC method (blind OOD detection)
python -m meanflow.evaluate_ood_denoising \
    --checkpoint checkpoints/model.pt \
    --data_path data/rml_denoising \
    --experiment_setting 1 \
    --method mdrc \
    --batch_size 512 \
    --device cuda

# Other blind methods also available
--method snr_improvement  # SNR-based
--method correlation      # Correlation-based
--method min_error        # SNR of outputs
--method avg_error        # Average SNR
--method improvement      # Consistency across classes
```

---

## Verification Checklist

- [x] All methods are blind (no clean signal used during scoring)
- [x] True labels NOT used during test-time OOD scoring
- [x] True labels ONLY used for: (1) ID validation statistics, (2) evaluation metrics
- [x] All methods can work in realistic deployment scenarios
- [x] MDRC method properly implements Mahalanobis distance with multi-domain features
- [x] Feature extraction uses only input-output residuals (blind)
- [x] Statistics computed from ID data using correct labels
- [x] Testing tries all classes blindly and uses best-case score

---

## Key Insight

**The fundamental principle**: 
- **ID samples**: Model knows how to denoise them → "normal" residual patterns
- **OOD samples**: Model doesn't know how to handle them → "abnormal" residual patterns

All methods measure some form of "normality" of the denoising process without access to ground truth.




