# Training Improvements for Mean Flow Modulation Classification

## Problem Analysis

Your training results showed several critical issues:

1. **Validation AUROC was NaN**: The validation set contains only known classes (no unknowns), making AUROC computation undefined
2. **FPR ≈ 1 and TKR ≈ 0**: The energy threshold was improperly tuned, rejecting almost all samples
3. **Poor test AUROC (0.33)**: With `lambda_neg=0`, the model never learned to separate known/unknown energies
4. **Energy distributions overlapped**: Known mean energy -2.122, Unknown mean energy -2.106 (difference only 0.016)

## Root Causes

1. **Validation set design**: By design, validation contains only known classes for proper model selection
2. **Threshold tuning bug**: The code tried to maximize F1 on validation where all samples are known (F1 always 0)
3. **Disabled negative energy loss**: Without `lambda_neg`, the model doesn't learn to push OOD samples toward less negative energies
4. **No energy separation mechanism**: Missing ranking loss to enforce margin between positive/negative samples

## Implemented Solutions

### 1. Fixed Validation AUROC Computation

```python
# Check if we have both known and unknown samples
has_unknown = unknown_mask.sum() > 0
has_known = known_mask.sum() > 0

if has_unknown and has_known:
    auroc = roc_auc_score(ood_labels, -all_energies)
    aupr = auc(recall, precision)
else:
    # No unknown samples (e.g., validation set) - AUROC/AUPR undefined
    auroc = np.nan
    aupr = np.nan
```

### 2. Proper Threshold Tuning

For validation set (no unknowns), use target FPR on known samples:

```python
if has_unknown:
    # Test set: maximize F1 score
    # ... existing F1 optimization ...
else:
    # Validation set: use target FPR on known samples
    target_fpr = getattr(args, 'target_fpr', 0.05)  # Default 5% FPR
    percentile = (1.0 - target_fpr) * 100
    energy_threshold = np.percentile(all_energies[known_mask], percentile)
```

### 3. Soft Ranking Loss Implementation

Added soft ranking loss based on the formula you provided:

```python
# L_rank = (1/β) * log(1 + exp(β * (E_p - E_n + Δ)))
if lambda_rank > 0 and x_neg is not None:
    # Compute pairwise ranking loss
    pos_energies_expanded = pos_energies.unsqueeze(1)  # [batch, 1]
    neg_energies_expanded = neg_energies.unsqueeze(0)  # [1, batch]
    
    # E_p - E_n + Δ (we want this to be negative)
    energy_diff = pos_energies_expanded - neg_energies_expanded + rank_margin
    
    # Soft ranking loss
    rank_loss = (1.0 / rank_beta) * torch.log(1 + torch.exp(rank_beta * energy_diff))
    rank_loss = rank_loss.mean()  # Average over all pairs
```

### 4. Updated Checkpoint Selection

Changed from AUROC to closed-set accuracy for validation:

```python
# Use closed_set_accuracy since no unknowns in validation
is_best = val_metrics['eval/closed_set_accuracy'] > best_metric
```

### 5. New Command Line Arguments

Added parameters for improved training:

```python
--lambda_rank: Weight for soft ranking loss (default: 0.1)
--rank_margin: Margin delta for soft ranking loss (default: 0.5)
--rank_beta: Beta parameter for soft ranking loss (default: 10.0)
--target_fpr: Target false positive rate for validation threshold (default: 0.05)
```

## Recommended Hyperparameters

Based on your issues, here are recommended settings:

```bash
# Loss weights
--lambda_rec 1.0      # Reconstruction loss (keep as baseline)
--lambda_arc 0.2      # Small ArcFace for class separation
--lambda_pos 0.3      # Positive energy loss
--lambda_neg 0.2      # IMPORTANT: Must be > 0 (was 0 in your run)
--lambda_rank 0.1     # Soft ranking loss for separation

# Energy margins
--margin_pos -2.5     # More negative target for known samples
--margin_neg -1.5     # Less negative target for unknown samples
--rank_margin 0.5     # Minimum separation between pos/neg energies

# Other parameters
--target_fpr 0.05     # 5% false positive rate on validation
--use_arcface         # Enable for better class separation
```

## Expected Improvements

After these changes, you should see:

1. **Validation metrics**: 
   - AUROC/AUPR will show as NaN (expected)
   - FPR ≈ 5% (matches target_fpr)
   - Higher closed-set accuracy

2. **Test metrics**:
   - AUROC > 0.7 (vs. 0.33 before)
   - FPR ≈ 5-10% (vs. 99.95% before)
   - TKR > 90% (vs. 0.05% before)
   - Better F1 score for OOD detection

3. **Energy distributions**:
   - Larger gap between known/unknown mean energies
   - Known energies more negative (< -2.2)
   - Unknown energies less negative (> -1.8)

## Running the Improved Training

Execute the test script:

```bash
bash meanflow/scripts/test_improved_training.sh
```

Or run directly with your preferred settings:

```bash
python -m meanflow.train_modulation \
    --experiment_setting 1 \
    --lambda_neg 0.2 \    # Critical: must be > 0
    --lambda_rank 0.1 \   # New: soft ranking loss
    --target_fpr 0.05 \   # New: proper validation threshold
    --use_arcface \       # Recommended for class separation
    # ... other parameters
```

## Monitoring Training

Watch these metrics during training:

1. `train/neg_energy_loss`: Should decrease but remain > 0
2. `train/rank_loss`: Should decrease, enforcing energy separation
3. `val/closed_set_accuracy`: Should increase (used for checkpointing)
4. Energy means: Gap between known/unknown should widen over epochs

## Debugging Tips

If results don't improve:

1. **Check energy distributions**: Plot histograms of known vs unknown energies
2. **Adjust margins**: If energies don't separate, try wider margins (e.g., margin_pos=-3.0, margin_neg=-1.0)
3. **Increase lambda_rank**: If separation is insufficient, increase to 0.2-0.3
4. **Monitor individual losses**: Ensure all loss components are contributing

## Summary

The key insight is that your model needs explicit supervision to separate energy distributions. With `lambda_neg=0`, it had no incentive to push negative samples toward higher (less negative) energies. The soft ranking loss provides additional pairwise supervision to maintain a margin between distributions.

These changes should significantly improve your OOD detection performance while maintaining or improving closed-set accuracy.

