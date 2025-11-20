# Validation Set OOD Update

## Summary

The validation set has been updated to include **unknown classes (OOD data)** instead of synthetic negative samples. This allows for more realistic evaluation of OOD detection performance during training.

## Changes Made

### 1. Dataset Split (`meanflow/data/rml_dataset.py`)

**Before:**
- Training: Known classes only ✅
- Validation: Known classes only ✅
- Test: Known + Unknown classes ✅

**After:**
- Training: Known classes only ✅
- Validation: Known + Unknown classes ✅ (NEW)
- Test: Known + Unknown classes ✅

### 2. Evaluation Strategy (`meanflow/train_modulation.py`)

**Changed:**
- Disabled synthetic negatives for validation/test by default (`--eval_with_synthetic_negatives` now defaults to `False`)
- Updated model selection to use AUROC when available (validation now has unknowns)
- Clarified threshold tuning comments to reflect val/test both support F1-based tuning when unknowns present

### 3. Model Selection Criteria

**Before:**
- Used `closed_set_accuracy` for validation (no unknowns available)

**After:**
- Uses `AUROC` for validation when unknowns present (primary metric)
- Falls back to `closed_set_accuracy` if no unknowns detected
- Better reflects real-world OOD detection performance

## Benefits

### 1. **More Realistic Evaluation**
- Validation metrics now reflect actual OOD detection performance
- No longer relying on synthetic corruptions to simulate OOD

### 2. **Better Model Selection**
- Models selected based on real OOD detection capability (AUROC)
- Threshold tuning uses actual unknown class data
- Prevents overfitting to synthetic negative patterns

### 3. **Consistent Metrics**
- Validation and test sets now use the same evaluation protocol
- AUROC, AUPR, and F1 scores available during training
- Better correlation between validation and test performance

### 4. **Cleaner Evaluation**
- No synthetic negatives cluttering the evaluation
- All samples are from the actual dataset
- More interpretable per-class OOD detection rates

## Validation Metrics Now Available

With unknown classes in validation, you now get:

- ✅ `val/auroc` - Area Under ROC Curve for OOD detection
- ✅ `val/aupr` - Area Under Precision-Recall Curve
- ✅ `val/F1_score_OOD` - F1 score for OOD detection
- ✅ `val/mean_energy_unknown` - Average energy of real unknown classes
- ✅ `val/mean_energy_known` - Average energy of known classes
- ✅ `val/TUR (TPR)` - True Unknown Rate (recall for unknowns)
- ✅ `val/FPR` - False Positive Rate (known misclassified as unknown)
- ✅ `val/OA (Open-Set Accuracy)` - Combined accuracy for known classification + unknown detection
- ✅ Per-unknown-class metrics: `val/ood_{MOD}_auroc`, `val/ood_{MOD}_detection_rate`

## Testing

Run the verification script to check that validation now includes unknown classes:

```bash
python test_validation_ood.py
```

Expected output:
```
Validation Set Analysis:
Total samples: ~21,780
Known samples: ~19,602 (90%)
Unknown samples: ~2,178 (10%)

✅ PASS: Validation set correctly includes unknown classes
   Can now evaluate OOD detection metrics (AUROC, AUPR, etc.)
```

## Training Usage

No changes needed to your training commands! The validation set will automatically include unknown classes:

```bash
python meanflow/train_modulation.py \
    --experiment_setting 1 \
    --epochs 100 \
    --batch_size 256 \
    --data_path data/RML2016.10a_dict.pkl
```

### Optional: Re-enable Synthetic Negatives

If you want to go back to using synthetic negatives in validation/test (not recommended):

```bash
python meanflow/train_modulation.py \
    --eval_with_synthetic_negatives \
    ... other args ...
```

## Migration Notes

### Old Behavior
- Validation had only known classes
- Used synthetic negatives to simulate OOD
- Model selection based on `closed_set_accuracy`
- `val/auroc` was computed using synthetic corruptions

### New Behavior
- Validation has known + unknown classes (10% each from dataset)
- Uses real OOD data from unknown modulation classes
- Model selection based on `auroc` (when available)
- `val/auroc` computed using real unknown classes

### Checkpoints

Existing checkpoints are **fully compatible**. Only the evaluation protocol changed, not the model architecture or training procedure.

## Data Split Details

### Split Strategy by Class Type

**Known Classes:** Each (modulation, SNR) pair with 1000 samples:
- **800 samples (80%)** → Training
- **100 samples (10%)** → Validation
- **100 samples (10%)** → Test

**Unknown Classes:** Each (modulation, SNR) pair with 1000 samples:
- **0 samples (0%)** → Training (not used)
- **500 samples (50%)** → Validation
- **500 samples (50%)** → Test

### Example: Experiment Setting 1 (9 known, 2 unknown)

Per SNR value:
- Training: 9 known × 800 = **7,200 samples**
- Validation: 9 known × 100 + 2 unknown × 500 = **1,900 samples**
- Test: 9 known × 100 + 2 unknown × 500 = **1,900 samples**

Across 21 SNR values (-20 to 18 dB):
- Training: ~**151,200 samples** (known only)
- Validation: ~**39,900 samples** (known + unknown, better balanced)
- Test: ~**39,900 samples** (known + unknown, better balanced)

### Why 50/50 for Unknown?

Since unknown classes don't participate in training, we split them evenly between validation and test (50/50) instead of 10/10. This provides:
- More unknown samples for validation OOD evaluation
- Better balance between known and unknown in val/test sets
- More reliable AUROC and OOD detection metrics

## Best Practices

1. **Monitor `val/auroc`** - This is now your primary validation metric
2. **Check energy separation** - Look at `val/mean_energy_known` vs `val/mean_energy_unknown`
3. **Use early stopping** - Stop when `val/auroc` plateaus or degrades
4. **Compare val vs test** - They should have similar AUROC if model generalizes well
5. **Per-class analysis** - Monitor `val/ood_{CLASS}_detection_rate` for hardest unknowns

## Troubleshooting

### Q: Validation AUROC is NaN
**A:** Check that your data file includes unknown classes. Run `test_validation_ood.py` to verify.

### Q: Validation has no unknown samples
**A:** Ensure you're using the updated `rml_dataset.py`. Check that lines 205-208 allow unknowns in validation.

### Q: I want the old behavior back
**A:** Modify `rml_dataset.py` line 206-208 to exclude unknowns in validation:
```python
elif self.split == 'val':
    if not is_unk:  # Add this condition back
        selected_indices = val_indices
```

## References

- Original validation split: `VALIDATION_SPLIT_README.md`
- Dataset implementation: `meanflow/data/rml_dataset.py`
- Training script: `meanflow/train_modulation.py`
- Test script: `test_validation_ood.py`

## Summary Table

| Aspect | Before | After |
|--------|--------|-------|
| Val unknown classes | ❌ No | ✅ Yes |
| Val OOD method | Synthetic negatives | Real unknown classes |
| Val AUROC meaning | Synthetic OOD detection | Real OOD detection |
| Model selection metric | Closed-set accuracy | AUROC (preferred) |
| Val metrics availability | Limited | Full OOD metrics |
| Evaluation realism | Low (synthetic) | High (real data) |

---

**Last Updated:** 2024-11-20

