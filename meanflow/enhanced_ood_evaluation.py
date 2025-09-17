"""
Enhanced OOD Evaluation with Per-Class Energy Thresholds

This module provides advanced OOD detection capabilities including:
1. Per-class energy thresholds for more precise OOD detection
2. Per-unknown-class OOD accuracy computation
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, f1_score
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PerClassEnergyThresholdDetector:
    """
    OOD detector with per-class energy thresholds
    """
    
    def __init__(self, num_classes: int, percentile: float = 95):
        """
        Args:
            num_classes: Number of known classes
            percentile: Percentile for initial threshold estimation (default: 95)
        """
        self.num_classes = num_classes
        self.percentile = percentile
        self.per_class_thresholds = None
        self.global_threshold = None
        self.combination_method = 'min'  # 'min', 'weighted', or 'adaptive'
        
    def fit_thresholds(
        self,
        class_energies: np.ndarray,
        labels: np.ndarray,
        is_unknown: np.ndarray,
        method: str = 'percentile'
    ):
        """
        Fit per-class energy thresholds using validation data
        
        Args:
            class_energies: Per-class energy scores [n_samples, n_classes]
            labels: True class labels (only valid for known samples)
            is_unknown: Boolean array indicating unknown samples
            method: Method for threshold estimation ('percentile', 'f1_optimize', 'roc_optimize')
        """
        known_mask = ~is_unknown
        known_energies = class_energies[known_mask]
        known_labels = labels[known_mask]
        
        self.per_class_thresholds = np.zeros(self.num_classes)
        
        for class_idx in range(self.num_classes):
            # Get energies for samples that truly belong to this class
            class_mask = known_labels == class_idx
            
            if class_mask.sum() == 0:
                # No samples for this class, use global threshold
                self.per_class_thresholds[class_idx] = np.percentile(
                    known_energies[:, class_idx], self.percentile
                )
                continue
            
            # Energy scores for this class on its own samples
            class_specific_energies = known_energies[class_mask, class_idx]
            
            if method == 'percentile':
                # Use percentile-based threshold
                self.per_class_thresholds[class_idx] = np.percentile(
                    class_specific_energies, self.percentile
                )
                
            elif method == 'f1_optimize':
                # Optimize F1 score for detecting OOD vs this specific class
                all_energies = class_energies[:, class_idx]
                
                # Create binary labels: 0 for this class, 1 for others/unknown
                binary_labels = np.ones(len(labels))
                binary_labels[known_mask & (labels == class_idx)] = 0
                
                # Find optimal threshold
                thresholds = np.percentile(all_energies, np.linspace(0, 100, 100))
                best_f1 = 0
                best_threshold = thresholds[0]
                
                for threshold in thresholds:
                    predictions = all_energies > threshold
                    f1 = f1_score(binary_labels, predictions)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                
                self.per_class_thresholds[class_idx] = best_threshold
                
            elif method == 'roc_optimize':
                # Optimize based on ROC curve (e.g., at specific FPR)
                all_energies = class_energies[:, class_idx]
                binary_labels = np.ones(len(labels))
                binary_labels[known_mask & (labels == class_idx)] = 0
                
                from sklearn.metrics import roc_curve
                fpr, tpr, thresholds = roc_curve(binary_labels, all_energies)
                
                # Find threshold at FPR = 0.05 (5% false positive rate)
                target_fpr = 0.05
                idx = np.argmin(np.abs(fpr - target_fpr))
                self.per_class_thresholds[class_idx] = thresholds[idx]
        
        logger.info(f"Per-class thresholds fitted: {self.per_class_thresholds}")
        
        # Also fit a global threshold for comparison
        self._fit_global_threshold(class_energies, is_unknown)
        
    def _fit_global_threshold(self, class_energies: np.ndarray, is_unknown: np.ndarray):
        """
        Fit a global threshold using the minimum energy across classes
        """
        # Compute minimum energy across classes
        min_energies = np.min(class_energies, axis=1)
        
        # Find optimal global threshold
        ood_labels = is_unknown.astype(int)
        thresholds = np.percentile(min_energies, np.linspace(0, 100, 100))
        
        best_f1 = 0
        best_threshold = thresholds[0]
        
        for threshold in thresholds:
            predictions = min_energies > threshold
            f1 = f1_score(ood_labels, predictions)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.global_threshold = best_threshold
        logger.info(f"Global threshold: {self.global_threshold:.4f} (F1: {best_f1:.4f})")
    
    def detect_ood(
        self,
        class_energies: np.ndarray,
        use_per_class: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect OOD samples using per-class or global thresholds
        
        Args:
            class_energies: Per-class energy scores [n_samples, n_classes]
            use_per_class: Whether to use per-class thresholds
            
        Returns:
            predictions: Predicted class labels (-1 for OOD)
            is_ood: Boolean array indicating OOD detection
            confidence_scores: Confidence scores for predictions
        """
        if self.per_class_thresholds is None:
            raise ValueError("Thresholds not fitted. Call fit_thresholds first.")
        
        # Find best class (minimum energy) for each sample
        min_energies, best_classes = np.min(class_energies, axis=1), np.argmin(class_energies, axis=1)
        
        if use_per_class:
            # Check if energy exceeds the threshold for the predicted class
            is_ood = np.zeros(len(class_energies), dtype=bool)
            
            for i, (best_class, energies) in enumerate(zip(best_classes, class_energies)):
                # Method 1: Check threshold for predicted class only
                if self.combination_method == 'min':
                    is_ood[i] = energies[best_class] > self.per_class_thresholds[best_class]
                
                # Method 2: Weighted combination of all class thresholds
                elif self.combination_method == 'weighted':
                    # Weight by inverse energy (closer classes have more weight)
                    weights = 1.0 / (energies + 1e-6)
                    weights = weights / weights.sum()
                    weighted_threshold = np.sum(weights * self.per_class_thresholds)
                    is_ood[i] = min_energies[i] > weighted_threshold
                
                # Method 3: Adaptive - require passing threshold for at least one class
                elif self.combination_method == 'adaptive':
                    passed_any = np.any(energies <= self.per_class_thresholds)
                    is_ood[i] = not passed_any
        else:
            # Use global threshold
            is_ood = min_energies > self.global_threshold
        
        # Create predictions
        predictions = best_classes.copy()
        predictions[is_ood] = -1
        
        # Compute confidence scores (inverse of energy, normalized)
        confidence_scores = 1.0 / (1.0 + min_energies)
        
        return predictions, is_ood, confidence_scores


def evaluate_per_unknown_class(
    model,
    test_loader,
    class_mapping: Dict[str, int],
    unknown_classes: List[str],
    device: torch.device
) -> Dict[str, float]:
    """
    Compute OOD detection accuracy for each unknown class separately
    
    Args:
        model: Trained model with compute_energy_score method
        test_loader: Test data loader
        class_mapping: Mapping from class names to indices
        unknown_classes: List of unknown class names
        device: Torch device
        
    Returns:
        Dictionary with per-unknown-class metrics
    """
    model.eval()
    
    # Storage for each unknown class
    unknown_class_data = {cls: {'energies': [], 'samples': 0} for cls in unknown_classes}
    known_energies = []
    
    # Collect energies
    with torch.no_grad():
        for pos_samples, neg_samples, labels, info in tqdm(test_loader, desc="Collecting energies"):
            signals = pos_samples.to(device)
            
            # Compute energy scores
            energy_scores = model.compute_energy_score(signals).cpu().numpy()
            
            # Separate by class
            is_unknown = info['is_unknown'].numpy()
            
            for i, (energy, is_unk) in enumerate(zip(energy_scores, is_unknown)):
                if is_unk:
                    # Need to determine which unknown class this is
                    # This requires access to original modulation labels
                    # For now, we'll use the label as indicator (you may need to modify based on your data)
                    # In practice, you'd need to track the original modulation type
                    pass  # This needs dataset modification to track original modulation
                else:
                    known_energies.append(energy)
    
    # Convert to arrays
    known_energies = np.array(known_energies)
    
    # Compute detection metrics for each unknown class
    metrics = {}
    
    for unknown_cls in unknown_classes:
        if unknown_class_data[unknown_cls]['samples'] == 0:
            continue
            
        unknown_energies = np.array(unknown_class_data[unknown_cls]['energies'])
        
        # Combine for AUROC computation
        all_energies = np.concatenate([known_energies, unknown_energies])
        all_labels = np.concatenate([
            np.zeros(len(known_energies)),  # Known = 0
            np.ones(len(unknown_energies))   # Unknown = 1
        ])
        
        # Compute AUROC for this specific unknown class
        auroc = roc_auc_score(all_labels, all_energies)
        
        # Compute AUPR
        precision, recall, _ = precision_recall_curve(all_labels, all_energies)
        aupr = auc(recall, precision)
        
        # Find optimal threshold for this unknown class
        thresholds = np.percentile(all_energies, np.linspace(0, 100, 100))
        best_f1 = 0
        best_threshold = thresholds[0]
        
        for threshold in thresholds:
            predictions = all_energies > threshold
            f1 = f1_score(all_labels, predictions)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Compute accuracy at optimal threshold
        predictions = all_energies > best_threshold
        accuracy = np.mean(predictions == all_labels)
        
        metrics[f'ood_{unknown_cls}_auroc'] = auroc
        metrics[f'ood_{unknown_cls}_aupr'] = aupr
        metrics[f'ood_{unknown_cls}_f1'] = best_f1
        metrics[f'ood_{unknown_cls}_accuracy'] = accuracy
        metrics[f'ood_{unknown_cls}_threshold'] = best_threshold
        metrics[f'ood_{unknown_cls}_samples'] = unknown_class_data[unknown_cls]['samples']
        
        logger.info(f"Unknown class {unknown_cls}: AUROC={auroc:.4f}, F1={best_f1:.4f}, Acc={accuracy:.4f}")
    
    return metrics


def enhanced_evaluation_with_per_class_thresholds(
    model,
    val_loader,
    test_loader,
    num_classes: int,
    device: torch.device,
    threshold_method: str = 'f1_optimize'
) -> Dict[str, float]:
    """
    Enhanced evaluation using per-class energy thresholds
    
    Args:
        model: Trained model
        val_loader: Validation loader for threshold fitting
        test_loader: Test loader for evaluation
        num_classes: Number of known classes
        device: Torch device
        threshold_method: Method for threshold estimation
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Step 1: Collect validation data for threshold fitting
    val_class_energies = []
    val_labels = []
    val_is_unknown = []
    
    with torch.no_grad():
        for pos_samples, _, labels, info in tqdm(val_loader, desc="Collecting validation data"):
            signals = pos_samples.to(device)
            
            # Get per-class energies
            class_energies = model.compute_energy_score(signals, return_per_class=True)
            
            val_class_energies.append(class_energies.cpu().numpy())
            val_labels.append(labels.cpu().numpy())
            val_is_unknown.append(info['is_unknown'].numpy())
    
    val_class_energies = np.concatenate(val_class_energies)
    val_labels = np.concatenate(val_labels)
    val_is_unknown = np.concatenate(val_is_unknown)
    
    # Step 2: Fit per-class thresholds
    detector = PerClassEnergyThresholdDetector(num_classes)
    detector.fit_thresholds(val_class_energies, val_labels, val_is_unknown, method=threshold_method)
    
    # Step 3: Evaluate on test set
    test_class_energies = []
    test_labels = []
    test_is_unknown = []
    
    with torch.no_grad():
        for pos_samples, _, labels, info in tqdm(test_loader, desc="Evaluating on test set"):
            signals = pos_samples.to(device)
            
            class_energies = model.compute_energy_score(signals, return_per_class=True)
            
            test_class_energies.append(class_energies.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
            test_is_unknown.append(info['is_unknown'].numpy())
    
    test_class_energies = np.concatenate(test_class_energies)
    test_labels = np.concatenate(test_labels)
    test_is_unknown = np.concatenate(test_is_unknown)
    
    # Evaluate with per-class thresholds
    predictions_per_class, is_ood_per_class, confidence_per_class = detector.detect_ood(
        test_class_energies, use_per_class=True
    )
    
    # Evaluate with global threshold
    predictions_global, is_ood_global, confidence_global = detector.detect_ood(
        test_class_energies, use_per_class=False
    )
    
    # Compute metrics for both approaches
    metrics = {}
    
    for name, predictions, is_ood in [
        ('per_class', predictions_per_class, is_ood_per_class),
        ('global', predictions_global, is_ood_global)
    ]:
        # Known samples accuracy
        known_mask = ~test_is_unknown
        if known_mask.sum() > 0:
            known_predictions = predictions[known_mask]
            known_labels_subset = test_labels[known_mask]
            closed_set_acc = np.mean(known_predictions == known_labels_subset)
        else:
            closed_set_acc = 0.0
        
        # OOD detection metrics
        ood_labels = test_is_unknown.astype(int)
        
        # AUROC and AUPR
        min_energies = np.min(test_class_energies, axis=1)
        auroc = roc_auc_score(ood_labels, min_energies)
        
        precision, recall, _ = precision_recall_curve(ood_labels, min_energies)
        aupr = auc(recall, precision)
        
        # F1 score for OOD detection
        f1 = f1_score(ood_labels, is_ood)
        
        # Open-set accuracy
        correct = 0
        total = len(predictions)
        
        for i, (pred, is_unk) in enumerate(zip(predictions, test_is_unknown)):
            if is_unk:
                # Unknown sample: correct if detected as OOD
                if pred == -1:
                    correct += 1
            else:
                # Known sample: correct if classified correctly
                if pred == test_labels[i]:
                    correct += 1
        
        open_set_acc = correct / total if total > 0 else 0.0
        
        metrics[f'{name}_closed_set_accuracy'] = closed_set_acc
        metrics[f'{name}_open_set_accuracy'] = open_set_acc
        metrics[f'{name}_auroc'] = auroc
        metrics[f'{name}_aupr'] = aupr
        metrics[f'{name}_f1_ood'] = f1
        
        logger.info(f"\n{name.upper()} Threshold Results:")
        logger.info(f"  Closed-set Accuracy: {closed_set_acc:.4f}")
        logger.info(f"  Open-set Accuracy: {open_set_acc:.4f}")
        logger.info(f"  AUROC: {auroc:.4f}")
        logger.info(f"  AUPR: {aupr:.4f}")
        logger.info(f"  F1 (OOD): {f1:.4f}")
    
    # Add threshold information to metrics
    metrics['per_class_thresholds'] = detector.per_class_thresholds.tolist()
    metrics['global_threshold'] = float(detector.global_threshold)
    
    return metrics


# Example usage function
def demo_usage():
    """
    Example of how to use the enhanced OOD evaluation
    """
    import sys
    sys.path.append('/Users/Axer/Desktop/py-meanflow')
    
    from meanflow.models.meanflow_modulation import MeanFlowModulation
    from meanflow.data.rml_dataset import get_rml_dataloaders
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model (example)
    model_args = {
        'num_classes': 11,  # Example for experiment setting 1
        'model_channels': 128,
        'channel_mult': [1, 2, 2, 2],
        'num_res_blocks': 2,
        'dropout': 0.1,
        'use_arcface': True,
        'energy_temperature': 1.0
    }
    
    model = MeanFlowModulation(**model_args).to(device)
    
    # Load data
    train_loader, val_loader, test_loader = get_rml_dataloaders(
        data_path='/path/to/data',
        experiment_setting=1,
        batch_size=64,
        num_workers=4
    )
    
    # Evaluate with per-class thresholds
    metrics = enhanced_evaluation_with_per_class_thresholds(
        model=model,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=11,
        device=device,
        threshold_method='f1_optimize'
    )
    
    print("\nEvaluation Results:")
    for key, value in metrics.items():
        if isinstance(value, list):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    demo_usage()
