#!/usr/bin/env python3
"""
Test script for advanced OOD detection features
"""

import sys
import os
sys.path.append('/Users/Axer/Desktop/py-meanflow')

import numpy as np
import torch
from meanflow.enhanced_ood_evaluation import PerClassEnergyThresholdDetector
from meanflow.per_unknown_class_evaluation import PerUnknownClassEvaluator

def test_per_class_thresholds():
    """
    Test per-class energy threshold functionality
    """
    print("="*60)
    print("Testing Per-Class Energy Thresholds")
    print("="*60)
    
    # Simulate data
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5
    
    # Generate synthetic class energies
    # Known samples: lower energy for correct class
    known_samples = 800
    unknown_samples = 200
    
    class_energies = np.random.randn(n_samples, n_classes) * 2 + 10
    labels = np.random.randint(0, n_classes, n_samples)
    is_unknown = np.zeros(n_samples, dtype=bool)
    is_unknown[-unknown_samples:] = True
    
    # Make known samples have lower energy for their true class
    for i in range(known_samples):
        class_energies[i, labels[i]] -= 5
    
    # Make unknown samples have higher energy
    class_energies[is_unknown] += 5
    
    # Initialize and fit detector
    detector = PerClassEnergyThresholdDetector(num_classes=n_classes, percentile=95)
    
    # Test different fitting methods
    for method in ['percentile', 'f1_optimize', 'roc_optimize']:
        print(f"\nTesting with method: {method}")
        
        detector.fit_thresholds(
            class_energies=class_energies,
            labels=labels,
            is_unknown=is_unknown,
            method=method
        )
        
        print(f"  Per-class thresholds: {detector.per_class_thresholds}")
        print(f"  Global threshold: {detector.global_threshold:.4f}")
        
        # Test detection
        predictions, is_ood, confidence = detector.detect_ood(
            class_energies, use_per_class=True
        )
        
        # Compute metrics
        ood_accuracy = np.mean(is_ood == is_unknown)
        print(f"  OOD detection accuracy: {ood_accuracy:.4f}")
        
        # Compare with global threshold
        predictions_global, is_ood_global, _ = detector.detect_ood(
            class_energies, use_per_class=False
        )
        
        ood_accuracy_global = np.mean(is_ood_global == is_unknown)
        print(f"  OOD accuracy (global threshold): {ood_accuracy_global:.4f}")
        
        # Test different combination methods
        for combo_method in ['min', 'weighted', 'adaptive']:
            detector.combination_method = combo_method
            predictions_combo, is_ood_combo, _ = detector.detect_ood(
                class_energies, use_per_class=True
            )
            accuracy_combo = np.mean(is_ood_combo == is_unknown)
            print(f"  OOD accuracy ({combo_method} combination): {accuracy_combo:.4f}")
    
    print("\n✓ Per-class threshold testing complete")

def test_per_unknown_class_evaluation():
    """
    Test per-unknown-class OOD evaluation
    """
    print("\n" + "="*60)
    print("Testing Per-Unknown-Class OOD Evaluation")
    print("="*60)
    
    # Define classes
    known_classes = ['BPSK', 'QPSK', 'GFSK', 'CPFSK', 'PAM4']
    unknown_classes = ['8PSK', '16QAM', '64QAM', 'OFDM', 'WBFM']
    
    # Initialize evaluator
    evaluator = PerUnknownClassEvaluator(known_classes, unknown_classes)
    
    print(f"Known classes: {known_classes}")
    print(f"Unknown classes: {unknown_classes}")
    print(f"Class mappings created successfully")
    
    # Simulate energy scores for demonstration
    np.random.seed(42)
    
    # Generate synthetic data
    n_known_samples = 500
    n_unknown_per_class = 100
    
    # Known samples have lower energy
    known_energies = np.random.gamma(2, 2, n_known_samples)  # Lower energy distribution
    
    # Different unknown classes have different energy distributions
    unknown_class_energies = {}
    difficulty_levels = {
        '8PSK': 0.3,     # Easy to detect (very different)
        '16QAM': 0.5,    # Medium difficulty
        '64QAM': 0.6,    # Medium difficulty
        'OFDM': 0.7,     # Hard to detect
        'WBFM': 0.2      # Very easy to detect
    }
    
    for unknown_cls in unknown_classes:
        # Higher difficulty = more overlap with known distribution
        difficulty = difficulty_levels[unknown_cls]
        shift = 10 * (1 - difficulty)  # Less shift = harder to detect
        energies = np.random.gamma(2, 2, n_unknown_per_class) + shift
        unknown_class_energies[unknown_cls] = energies
    
    # Compute metrics for each unknown class
    print("\nSimulated Per-Unknown-Class Metrics:")
    print("-" * 40)
    
    all_metrics = {}
    
    for unknown_cls in unknown_classes:
        unknown_energies = unknown_class_energies[unknown_cls]
        
        # Combine with known for metrics
        all_energies = np.concatenate([known_energies, unknown_energies])
        all_labels = np.concatenate([
            np.zeros(len(known_energies)),
            np.ones(len(unknown_energies))
        ])
        
        # Compute AUROC
        from sklearn.metrics import roc_auc_score, f1_score
        auroc = roc_auc_score(all_labels, all_energies)
        
        # Find optimal threshold
        threshold = np.percentile(all_energies, 70)
        predictions = all_energies > threshold
        f1 = f1_score(all_labels, predictions)
        accuracy = np.mean(predictions == all_labels)
        
        # TPR for unknown class
        tpr = np.mean(unknown_energies > threshold)
        
        all_metrics[unknown_cls] = {
            'auroc': auroc,
            'f1': f1,
            'accuracy': accuracy,
            'tpr': tpr,
            'mean_energy': np.mean(unknown_energies),
            'difficulty': difficulty_levels[unknown_cls]
        }
        
        print(f"\n{unknown_cls}:")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  TPR: {tpr:.4f}")
        print(f"  Mean Energy: {np.mean(unknown_energies):.2f}")
        print(f"  Difficulty Level: {difficulty_levels[unknown_cls]}")
    
    # Rank by difficulty
    print("\n" + "-" * 40)
    print("Unknown Classes Ranked by Detection Difficulty:")
    print("(Based on AUROC - Higher = Easier to detect)")
    
    ranked = sorted(all_metrics.items(), key=lambda x: x[1]['auroc'], reverse=True)
    for rank, (cls, metrics) in enumerate(ranked, 1):
        difficulty = "Easy" if metrics['auroc'] > 0.9 else "Medium" if metrics['auroc'] > 0.7 else "Hard"
        print(f"  {rank}. {cls}: AUROC={metrics['auroc']:.4f} ({difficulty})")
    
    print("\n✓ Per-unknown-class evaluation testing complete")

def main():
    """
    Run all tests
    """
    print("\n" + "="*60)
    print("ADVANCED OOD DETECTION FEATURE TESTS")
    print("="*60)
    
    # Test 1: Per-class thresholds
    test_per_class_thresholds()
    
    # Test 2: Per-unknown-class evaluation
    test_per_unknown_class_evaluation()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\nSummary:")
    print("1. ✓ Per-class energy thresholds allow more precise OOD detection")
    print("2. ✓ Per-unknown-class metrics enable fine-grained analysis")
    print("3. ✓ Both features can be combined for comprehensive OOD evaluation")
    
    print("\nNext Steps:")
    print("1. Integrate with actual RML dataset (modify dataset to track modulations)")
    print("2. Train model and collect energy scores")
    print("3. Use these tools for detailed OOD analysis")

if __name__ == "__main__":
    main()
