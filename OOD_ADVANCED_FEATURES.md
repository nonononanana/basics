# Advanced OOD Detection Features

This document explains the advanced OOD (Out-of-Distribution) detection features, specifically answering:
1. How to set per-class energy thresholds
2. How to compute OOD accuracy for each unknown class

## Question 1: Per-Class Energy Thresholds

### Why Per-Class Thresholds?

Different known classes may have different energy distributions. Using a single global threshold might not be optimal for all classes. Per-class thresholds allow more precise OOD detection by considering the specific energy characteristics of each class.

### Implementation

The `PerClassEnergyThresholdDetector` in `enhanced_ood_evaluation.py` provides this functionality:

```python
from meanflow.enhanced_ood_evaluation import PerClassEnergyThresholdDetector

# Initialize detector
detector = PerClassEnergyThresholdDetector(num_classes=11)

# Fit thresholds on validation data
detector.fit_thresholds(
    class_energies=val_class_energies,  # [n_samples, n_classes]
    labels=val_labels,                   # True labels for known samples
    is_unknown=val_is_unknown,           # Boolean array
    method='f1_optimize'                 # or 'percentile', 'roc_optimize'
)

# Detect OOD with per-class thresholds
predictions, is_ood, confidence = detector.detect_ood(
    class_energies=test_class_energies,
    use_per_class=True  # Set to False for global threshold
)
```

### Threshold Setting Methods

1. **Percentile-based** (`method='percentile'`):
   - Sets threshold at a specific percentile (e.g., 95th) of the energy distribution
   - Simple and robust, good for initial exploration

2. **F1-optimization** (`method='f1_optimize'`):
   - Finds threshold that maximizes F1 score for OOD detection
   - Balances precision and recall

3. **ROC-optimization** (`method='roc_optimize'`):
   - Finds threshold at a specific false positive rate (e.g., 5%)
   - Good when you have a target FPR requirement

### Overall Energy Threshold with Per-Class Thresholds

When using per-class thresholds, the overall OOD detection works as follows:

1. **Method 1: Minimum Energy** (default)
   ```python
   # For each sample:
   # 1. Find the best matching class (minimum energy)
   # 2. Check if energy exceeds that class's threshold
   best_class = argmin(class_energies)
   is_ood = class_energies[best_class] > per_class_thresholds[best_class]
   ```

2. **Method 2: Weighted Combination**
   ```python
   # Weight thresholds by inverse energy (closer classes have more weight)
   weights = 1.0 / (class_energies + epsilon)
   weighted_threshold = sum(weights * per_class_thresholds) / sum(weights)
   is_ood = min(class_energies) > weighted_threshold
   ```

3. **Method 3: Adaptive**
   ```python
   # Require passing at least one class threshold
   passed_any = any(class_energies <= per_class_thresholds)
   is_ood = not passed_any
   ```

### Example Usage

```python
# Complete example
from meanflow.enhanced_ood_evaluation import enhanced_evaluation_with_per_class_thresholds

metrics = enhanced_evaluation_with_per_class_thresholds(
    model=model,
    val_loader=val_loader,    # For fitting thresholds
    test_loader=test_loader,   # For evaluation
    num_classes=11,
    device=device,
    threshold_method='f1_optimize'
)

# Results include both per-class and global threshold performance
print(f"Per-class Open-Set Accuracy: {metrics['per_class_open_set_accuracy']:.4f}")
print(f"Global Open-Set Accuracy: {metrics['global_open_set_accuracy']:.4f}")
print(f"Per-class Thresholds: {metrics['per_class_thresholds']}")
```

## Question 2: OOD Accuracy for Each Unknown Class

### Why Per-Unknown-Class Metrics?

Different unknown classes may have varying difficulty levels for OOD detection. Some might be very different from known classes (easy to detect), while others might be similar (hard to detect). Computing metrics for each unknown class helps identify:
- Which unknown modulations are problematic
- Whether certain unknown classes are consistently misclassified as specific known classes
- The overall robustness of the OOD detection system

### Implementation

The `PerUnknownClassEvaluator` in `per_unknown_class_evaluation.py` provides this functionality:

```python
from meanflow.per_unknown_class_evaluation import PerUnknownClassEvaluator

# Initialize evaluator
evaluator = PerUnknownClassEvaluator(
    known_classes=['BPSK', 'QPSK', ...],    # List of known modulations
    unknown_classes=['8PSK', '16QAM', ...]   # List of unknown modulations
)

# Evaluate each unknown class
metrics = evaluator.evaluate_per_unknown_class(
    model=model,
    test_loader=test_loader,  # Must provide original modulation info
    device=device,
    energy_threshold=threshold  # Optional, will compute if None
)
```

### Metrics Computed for Each Unknown Class

For each unknown modulation type, the system computes:

1. **AUROC**: Area under ROC curve for detecting this specific unknown class vs all known classes
2. **AUPR**: Area under Precision-Recall curve
3. **Accuracy**: Detection accuracy at the energy threshold
4. **F1 Score**: Balanced metric for detection performance
5. **TPR**: True Positive Rate (sensitivity) - what fraction of this unknown class is correctly detected as OOD
6. **Mean Energy**: Average energy score for this unknown class
7. **Optimal Threshold**: Best threshold specifically for this unknown class

### Example Output

```
Per-Unknown-Class OOD Detection Metrics:

  8PSK:
    Samples: 1000
    AUROC: 0.9234
    AUPR: 0.9156
    Accuracy (global thresh): 0.8765
    F1 (global thresh): 0.8534
    Accuracy (optimal thresh): 0.9012
    F1 (optimal thresh): 0.8923
    TPR: 0.8450
    Mean Energy: 15.234 ± 3.456

  16QAM:
    Samples: 1000
    AUROC: 0.7654
    AUPR: 0.7234
    Accuracy (global thresh): 0.7123
    F1 (global thresh): 0.6934
    Accuracy (optimal thresh): 0.7534
    F1 (optimal thresh): 0.7234
    TPR: 0.6780
    Mean Energy: 12.456 ± 4.123

Unknown Classes Ranked by Detection Difficulty:
(Higher AUROC = Easier to detect as OOD)
  1. 8PSK: AUROC=0.9234 (Easy)
  2. 64QAM: AUROC=0.8923 (Easy)
  3. 16QAM: AUROC=0.7654 (Medium)
  4. PAM4: AUROC=0.6234 (Hard)
```

### Data Requirements

To compute per-unknown-class metrics, you need to track the original modulation type for each sample. This requires modifying the dataset to provide this information:

```python
# Modified dataset __getitem__ method
def __getitem__(self, idx):
    pos_sample, neg_sample, label, info = self.original_getitem(idx)
    
    # Add original modulation type
    original_modulation = self.get_modulation_type(idx)
    info['original_modulation'] = original_modulation
    
    return pos_sample, neg_sample, label, info
```

### Detection Heatmap

The system can also create a heatmap showing how each unknown class is classified:

```python
# Create detection heatmap
heatmap = evaluator.create_detection_heatmap(
    model=model,
    test_loader=test_loader,
    device=device,
    energy_threshold=threshold
)

# Heatmap shows:
# Rows: Unknown classes
# Columns: Known class predictions + correct OOD detection
# Values: Percentage of samples
```

## Practical Recommendations

### 1. Choosing Threshold Methods

- **For balanced performance**: Use F1-optimization
- **For conservative detection** (minimize false positives): Use high percentile (e.g., 99th)
- **For aggressive detection** (minimize false negatives): Use lower percentile (e.g., 90th)
- **For specific requirements**: Use ROC-optimization with target FPR

### 2. When to Use Per-Class vs Global Thresholds

**Use per-class thresholds when:**
- Known classes have significantly different energy distributions
- Some classes are inherently more variable than others
- You need maximum detection accuracy

**Use global threshold when:**
- Simplicity is important
- Known classes have similar energy distributions
- You have limited validation data

### 3. Interpreting Per-Unknown-Class Results

- **High AUROC (>0.9)**: Unknown class is very different from known classes, easy to detect
- **Medium AUROC (0.7-0.9)**: Moderate difficulty, may share some characteristics with known classes
- **Low AUROC (<0.7)**: Hard to detect, very similar to some known classes

### 4. Improving Detection for Specific Unknown Classes

If certain unknown classes have poor detection:

1. **Analyze confusion patterns**: Which known classes are they confused with?
2. **Adjust class-specific thresholds**: Lower threshold for known classes that are often confused
3. **Consider feature engineering**: Add features that better distinguish problematic pairs
4. **Use ensemble methods**: Combine multiple models with different strengths

## Integration Example

```python
# Complete integration example
import torch
from meanflow.models.meanflow_modulation import MeanFlowModulation
from meanflow.enhanced_ood_evaluation import PerClassEnergyThresholdDetector
from meanflow.per_unknown_class_evaluation import PerUnknownClassEvaluator

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MeanFlowModulation(...).to(device)

# 1. Fit per-class thresholds on validation set
detector = PerClassEnergyThresholdDetector(num_classes=11)
detector.fit_thresholds(val_energies, val_labels, val_is_unknown, method='f1_optimize')

# 2. Evaluate with per-class thresholds on test set
test_predictions, test_is_ood, test_confidence = detector.detect_ood(
    test_energies, use_per_class=True
)

# 3. Compute per-unknown-class metrics
evaluator = PerUnknownClassEvaluator(known_classes, unknown_classes)
unknown_metrics = evaluator.evaluate_per_unknown_class(
    model, test_loader, device, detector.global_threshold
)

# 4. Analyze results
print(f"Per-class thresholds: {detector.per_class_thresholds}")
print(f"Hardest unknown class: {unknown_metrics['difficulty_ranking'][-1]}")
print(f"Easiest unknown class: {unknown_metrics['difficulty_ranking'][0]}")
```

## Summary

1. **Per-class energy thresholds** provide more precise OOD detection by considering each known class's specific energy distribution. The overall threshold is determined by the combination method (minimum, weighted, or adaptive).

2. **Per-unknown-class OOD accuracy** allows fine-grained analysis of which unknown modulations are easier or harder to detect, enabling targeted improvements to the OOD detection system.

Both features work together to provide a comprehensive understanding of the OOD detection performance and identify areas for improvement.
