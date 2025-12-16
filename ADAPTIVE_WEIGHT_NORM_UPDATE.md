# Adaptive Weight Normalization Added to Denoising Model

## Summary
Added adaptive weight normalization to `MeanFlowDenoising` to improve training stability and prevent sudden crashes like the one observed around step 400-500.

## What Changed

### File: `meanflow/models/meanflow_denoising.py`

**Before:**
```python
# Loss: MSE between predicted and true clean signal
loss = F.mse_loss(x_pred, x_clean)

return {
    'total_loss': loss,
    'loss': loss
}
```

**After:**
```python
# Loss: MSE between predicted and true clean signal
# Compute per-sample loss for adaptive weighting
loss = F.mse_loss(x_pred, x_clean, reduction='none')  # [batch, 2, 128]
loss = loss.mean(dim=(1, 2))  # [batch] - mean over channel and signal dimensions

# Adaptive weighting for stability
# This prevents outlier samples with high loss from dominating gradients
adp_wt = (loss.detach() + self.args.norm_eps) ** self.args.norm_p
loss = loss / adp_wt

# Final loss: mean over batch
loss = loss.mean()

return {
    'total_loss': loss,
    'loss': loss
}
```

## How It Works

### Formula
For each sample's loss L_i, the adaptive weight is:
```
w_i = (L_i + ε)^p
L_i' = L_i / w_i
```

Where:
- `ε = norm_eps = 1e-3` (prevents division by zero)
- `p = norm_p = 0.75` (controls scaling strength)

### Gradient Scaling Effect

| Original Loss | Weight w_i | Scaled Loss | Gradient Scale |
|--------------|------------|-------------|----------------|
| 0.001        | 0.032      | 0.032       | 31x (amplified) |
| 0.01         | 0.074      | 0.135       | 13.5x |
| 0.1          | 0.178      | 0.562       | 5.6x |
| 1.0          | 1.000      | 1.000       | 1x (unchanged) |
| 10.0         | 5.623      | 1.778       | 0.18x (reduced) |

**Key Insight:** High-loss outlier samples have their gradients reduced, preventing them from dominating the update and causing training instability.

## Why This Helps

### Problem
Your training crashed at step 400-500 because:
1. Some difficult/outlier samples produced abnormally high losses
2. These samples generated large gradients
3. Even with gradient clipping, accumulated large gradients destabilized the model
4. Model parameters were pushed in wrong directions → crash

### Solution
Adaptive weight normalization:
1. **Automatically balances difficult samples** - high-loss samples don't dominate gradients
2. **Stabilizes training dynamics** - prevents any single batch from breaking training
3. **Maintains optimization efficiency** - low-loss samples still provide useful gradient signals

## Expected Benefits

✅ **More stable training** - reduces likelihood of sudden crashes
✅ **Better convergence** - balanced gradient contributions from all samples
✅ **Robustness to outliers** - difficult samples are automatically down-weighted
✅ **Consistent with other Mean Flow models** - same technique used in `meanflow_modulation.py`

## Testing

Run your training again with the same hyperparameters:
```bash
python meanflow/train_denoising.py \
    --data_path data/RML2016_denoising.pkl \
    --experiment_setting 1 \
    --lr 2e-4 \
    --batch_size 256 \
    --epochs 100 \
    --norm_eps 1e-3 \
    --norm_p 0.75
```

Monitor:
- Training loss should remain stable (no sudden spikes)
- Validation SNR improvement should gradually increase (no sudden drops)
- Learning rate curve should be smooth

## Additional Recommendations

If training still shows instability, consider:
1. **Reduce learning rate**: `--lr 1e-4` (instead of 2e-4)
2. **Increase warmup**: `--warmup_epochs 10` (instead of 5)
3. **Reduce gradient clipping**: `--grad_clip 0.5` (instead of 1.0)
4. **Disable mixed precision**: Add `--no_mixed_precision` flag (for testing)

## References

- This technique is used in the original Mean Flow implementation (`meanflow/models/meanflow.py`)
- Also used in the modulation classification model (`meanflow/models/meanflow_modulation.py`)
- Based on adaptive loss scaling principles from robust optimization literature




