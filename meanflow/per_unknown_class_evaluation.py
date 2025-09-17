"""
Per-Unknown-Class OOD Evaluation

This module provides functionality to compute OOD detection metrics
for each unknown class separately, allowing for fine-grained analysis
of which unknown classes are easier or harder to detect.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, f1_score, confusion_matrix
import logging
from tqdm import tqdm
from collections import defaultdict

logger = logging.getLogger(__name__)


class PerUnknownClassEvaluator:
    """
    Evaluator that computes OOD detection metrics for each unknown class separately
    """
    
    def __init__(self, known_classes: List[str], unknown_classes: List[str]):
        """
        Args:
            known_classes: List of known class names
            unknown_classes: List of unknown class names
        """
        self.known_classes = known_classes
        self.unknown_classes = unknown_classes
        self.num_known = len(known_classes)
        self.num_unknown = len(unknown_classes)
        
        # Create mappings
        self.known_to_idx = {cls: idx for idx, cls in enumerate(known_classes)}
        self.unknown_to_idx = {cls: idx for idx, cls in enumerate(unknown_classes)}
        
    def evaluate_per_unknown_class(
        self,
        model,
        test_loader,
        device: torch.device,
        energy_threshold: float = None
    ) -> Dict[str, float]:
        """
        Evaluate OOD detection performance for each unknown class separately
        
        Args:
            model: Trained model with compute_energy_score method
            test_loader: Modified test loader that provides original modulation info
            device: Torch device
            energy_threshold: Energy threshold for OOD detection (if None, will compute optimal)
            
        Returns:
            Dictionary with per-unknown-class metrics
        """
        model.eval()
        
        # Storage for energies by class
        known_energies = []
        known_labels = []
        unknown_class_energies = defaultdict(list)
        
        # Collect energies
        logger.info("Collecting energy scores for all samples...")
        
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc="Computing energies"):
                # Handle different data formats
                if len(batch_data) == 4:
                    pos_samples, _, labels, info = batch_data
                    original_modulations = info.get('original_modulation', None)
                else:
                    # Extended format with original modulation
                    pos_samples, _, labels, info, original_modulations = batch_data
                
                signals = pos_samples.to(device)
                
                # Compute energy scores
                energy_scores = model.compute_energy_score(signals).cpu().numpy()
                
                # Separate by class type
                is_unknown = info['is_unknown'].numpy()
                
                for i, (energy, is_unk) in enumerate(zip(energy_scores, is_unknown)):
                    if is_unk:
                        # Get the original modulation type for unknown samples
                        if original_modulations is not None:
                            mod_type = original_modulations[i]
                            if mod_type in self.unknown_classes:
                                unknown_class_energies[mod_type].append(energy)
                    else:
                        known_energies.append(energy)
                        known_labels.append(labels[i].item())
        
        # Convert to arrays
        known_energies = np.array(known_energies)
        
        # If no threshold provided, compute optimal global threshold
        if energy_threshold is None:
            all_unknown_energies = []
            for energies in unknown_class_energies.values():
                all_unknown_energies.extend(energies)
            all_unknown_energies = np.array(all_unknown_energies)
            
            if len(all_unknown_energies) > 0:
                # Find threshold that maximizes F1 score
                all_energies = np.concatenate([known_energies, all_unknown_energies])
                all_labels = np.concatenate([
                    np.zeros(len(known_energies)),
                    np.ones(len(all_unknown_energies))
                ])
                
                energy_threshold = self._find_optimal_threshold(all_energies, all_labels)
            else:
                energy_threshold = np.percentile(known_energies, 95)
        
        logger.info(f"Using energy threshold: {energy_threshold:.4f}")
        
        # Compute metrics for each unknown class
        metrics = {}
        metrics['global_energy_threshold'] = energy_threshold
        
        # Global metrics first
        if len(unknown_class_energies) > 0:
            all_unknown_energies = []
            for energies in unknown_class_energies.values():
                all_unknown_energies.extend(energies)
            all_unknown_energies = np.array(all_unknown_energies)
            
            # Global AUROC
            all_energies = np.concatenate([known_energies, all_unknown_energies])
            all_labels = np.concatenate([
                np.zeros(len(known_energies)),
                np.ones(len(all_unknown_energies))
            ])
            
            global_auroc = roc_auc_score(all_labels, all_energies)
            metrics['global_auroc'] = global_auroc
            
            # Global detection accuracy at threshold
            predictions = all_energies > energy_threshold
            global_accuracy = np.mean(predictions == all_labels)
            metrics['global_ood_accuracy'] = global_accuracy
            
            logger.info(f"\nGlobal OOD Detection Metrics:")
            logger.info(f"  AUROC: {global_auroc:.4f}")
            logger.info(f"  Accuracy: {global_accuracy:.4f}")
        
        # Per-unknown-class metrics
        logger.info(f"\nPer-Unknown-Class OOD Detection Metrics:")
        
        for unknown_cls in self.unknown_classes:
            if unknown_cls not in unknown_class_energies or len(unknown_class_energies[unknown_cls]) == 0:
                logger.warning(f"  No samples found for unknown class: {unknown_cls}")
                continue
            
            unknown_energies = np.array(unknown_class_energies[unknown_cls])
            
            # Combine known and this specific unknown class
            combined_energies = np.concatenate([known_energies, unknown_energies])
            combined_labels = np.concatenate([
                np.zeros(len(known_energies)),  # Known = 0
                np.ones(len(unknown_energies))   # Unknown = 1
            ])
            
            # AUROC for this specific unknown class
            auroc = roc_auc_score(combined_labels, combined_energies)
            
            # AUPR
            precision, recall, _ = precision_recall_curve(combined_labels, combined_energies)
            aupr = auc(recall, precision)
            
            # Find optimal threshold for this specific unknown class
            class_specific_threshold = self._find_optimal_threshold(
                combined_energies, combined_labels
            )
            
            # Detection metrics at global threshold
            predictions_global = combined_energies > energy_threshold
            accuracy_global = np.mean(predictions_global == combined_labels)
            f1_global = f1_score(combined_labels, predictions_global)
            
            # Detection metrics at class-specific threshold
            predictions_specific = combined_energies > class_specific_threshold
            accuracy_specific = np.mean(predictions_specific == combined_labels)
            f1_specific = f1_score(combined_labels, predictions_specific)
            
            # True Positive Rate (sensitivity) - correctly detected as unknown
            tpr = np.mean(unknown_energies > energy_threshold)
            
            # False Positive Rate - incorrectly detected as unknown (from known samples)
            fpr = np.mean(known_energies > energy_threshold)
            
            # Store metrics
            metrics[f'{unknown_cls}_auroc'] = auroc
            metrics[f'{unknown_cls}_aupr'] = aupr
            metrics[f'{unknown_cls}_accuracy_global_thresh'] = accuracy_global
            metrics[f'{unknown_cls}_f1_global_thresh'] = f1_global
            metrics[f'{unknown_cls}_accuracy_specific_thresh'] = accuracy_specific
            metrics[f'{unknown_cls}_f1_specific_thresh'] = f1_specific
            metrics[f'{unknown_cls}_optimal_threshold'] = class_specific_threshold
            metrics[f'{unknown_cls}_tpr'] = tpr
            metrics[f'{unknown_cls}_mean_energy'] = np.mean(unknown_energies)
            metrics[f'{unknown_cls}_std_energy'] = np.std(unknown_energies)
            metrics[f'{unknown_cls}_num_samples'] = len(unknown_energies)
            
            logger.info(f"\n  {unknown_cls}:")
            logger.info(f"    Samples: {len(unknown_energies)}")
            logger.info(f"    AUROC: {auroc:.4f}")
            logger.info(f"    AUPR: {aupr:.4f}")
            logger.info(f"    Accuracy (global thresh): {accuracy_global:.4f}")
            logger.info(f"    F1 (global thresh): {f1_global:.4f}")
            logger.info(f"    Accuracy (optimal thresh): {accuracy_specific:.4f}")
            logger.info(f"    F1 (optimal thresh): {f1_specific:.4f}")
            logger.info(f"    TPR: {tpr:.4f}")
            logger.info(f"    Mean Energy: {np.mean(unknown_energies):.4f} ± {np.std(unknown_energies):.4f}")
        
        # Compute ranking of unknown classes by difficulty
        self._compute_difficulty_ranking(metrics)
        
        return metrics
    
    def _find_optimal_threshold(self, energies: np.ndarray, labels: np.ndarray) -> float:
        """
        Find optimal energy threshold that maximizes F1 score
        """
        thresholds = np.percentile(energies, np.linspace(0, 100, 100))
        best_f1 = 0
        best_threshold = thresholds[0]
        
        for threshold in thresholds:
            predictions = energies > threshold
            f1 = f1_score(labels, predictions)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold
    
    def _compute_difficulty_ranking(self, metrics: Dict[str, float]):
        """
        Rank unknown classes by detection difficulty (based on AUROC)
        """
        class_aurocs = []
        
        for unknown_cls in self.unknown_classes:
            auroc_key = f'{unknown_cls}_auroc'
            if auroc_key in metrics:
                class_aurocs.append((unknown_cls, metrics[auroc_key]))
        
        # Sort by AUROC (higher = easier to detect)
        class_aurocs.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("\nUnknown Classes Ranked by Detection Difficulty:")
        logger.info("(Higher AUROC = Easier to detect as OOD)")
        
        for rank, (cls, auroc) in enumerate(class_aurocs, 1):
            difficulty = "Easy" if auroc > 0.9 else "Medium" if auroc > 0.7 else "Hard"
            logger.info(f"  {rank}. {cls}: AUROC={auroc:.4f} ({difficulty})")
        
        # Add to metrics
        metrics['difficulty_ranking'] = [cls for cls, _ in class_aurocs]
        metrics['difficulty_scores'] = [auroc for _, auroc in class_aurocs]
    
    def create_detection_heatmap(
        self,
        model,
        test_loader,
        device: torch.device,
        energy_threshold: float
    ) -> np.ndarray:
        """
        Create a heatmap showing detection rates for each unknown class
        versus each known class prediction
        
        Returns:
            Heatmap matrix [num_unknown, num_known + 1] where the last column
            represents correct OOD detection
        """
        model.eval()
        
        # Initialize confusion matrix
        # Rows: unknown classes, Cols: known class predictions + OOD detection
        heatmap = np.zeros((self.num_unknown, self.num_known + 1))
        unknown_counts = np.zeros(self.num_unknown)
        
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc="Creating detection heatmap"):
                if len(batch_data) == 5:
                    pos_samples, _, labels, info, original_modulations = batch_data
                else:
                    pos_samples, _, labels, info = batch_data
                    original_modulations = info.get('original_modulation', None)
                
                if original_modulations is None:
                    continue
                
                signals = pos_samples.to(device)
                is_unknown = info['is_unknown'].numpy()
                
                # Get predictions
                predictions, energy_scores, class_energies = model.classify_with_rejection(
                    signals, energy_threshold
                )
                
                predictions = predictions.cpu().numpy()
                
                # Process unknown samples
                unknown_mask = is_unknown
                if unknown_mask.sum() == 0:
                    continue
                
                unknown_preds = predictions[unknown_mask]
                unknown_mods = [original_modulations[i] for i in range(len(is_unknown)) if is_unknown[i]]
                
                for pred, mod in zip(unknown_preds, unknown_mods):
                    if mod not in self.unknown_to_idx:
                        continue
                    
                    unknown_idx = self.unknown_to_idx[mod]
                    unknown_counts[unknown_idx] += 1
                    
                    if pred == -1:
                        # Correctly detected as OOD
                        heatmap[unknown_idx, -1] += 1
                    elif pred >= 0 and pred < self.num_known:
                        # Misclassified as known class
                        heatmap[unknown_idx, pred] += 1
        
        # Normalize to percentages
        for i in range(self.num_unknown):
            if unknown_counts[i] > 0:
                heatmap[i] = heatmap[i] / unknown_counts[i] * 100
        
        return heatmap


# Modified dataset class that tracks original modulation
class EnhancedRMLDataset(torch.utils.data.Dataset):
    """
    Enhanced RML dataset that tracks original modulation types for unknown samples
    """
    
    def __init__(self, original_dataset):
        """
        Wrap the original dataset to add modulation tracking
        """
        self.dataset = original_dataset
        
        # Get modulation type for each sample
        self.original_modulations = []
        
        # This needs to be implemented based on how your dataset stores the data
        # For example, if the dataset has a mapping from indices to modulation types
        for idx in range(len(self.dataset)):
            # You'll need to extract the original modulation type
            # This is dataset-specific implementation
            mod_type = self._get_modulation_type(idx)
            self.original_modulations.append(mod_type)
    
    def _get_modulation_type(self, idx):
        """
        Get the original modulation type for a sample
        This needs to be implemented based on your dataset structure
        """
        # Placeholder - implement based on your dataset
        # For example, you might have a list or dict mapping indices to modulation types
        return "UNKNOWN"  # Replace with actual implementation
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get original data
        pos_sample, neg_sample, label, info = self.dataset[idx]
        
        # Add original modulation to output
        original_mod = self.original_modulations[idx]
        
        return pos_sample, neg_sample, label, info, original_mod


def demo_per_unknown_evaluation():
    """
    Demonstration of per-unknown-class evaluation
    """
    import sys
    sys.path.append('/Users/Axer/Desktop/py-meanflow')
    
    from meanflow.models.meanflow_modulation import MeanFlowModulation
    from meanflow.data.rml_dataset import get_rml_dataloaders, EXPERIMENT_SETTINGS
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_setting = 1
    
    # Get class configuration
    known_classes = EXPERIMENT_SETTINGS[experiment_setting]['known']
    unknown_classes = EXPERIMENT_SETTINGS[experiment_setting]['unknown']
    
    logger.info(f"Known classes: {known_classes}")
    logger.info(f"Unknown classes: {unknown_classes}")
    
    # Initialize evaluator
    evaluator = PerUnknownClassEvaluator(known_classes, unknown_classes)
    
    # Load model
    model_args = {
        'num_classes': len(known_classes),
        'model_channels': 128,
        'channel_mult': [1, 2, 2, 2],
        'num_res_blocks': 2,
        'dropout': 0.1,
        'use_arcface': True,
        'energy_temperature': 1.0
    }
    
    model = MeanFlowModulation(**model_args).to(device)
    
    # Load checkpoint if available
    checkpoint_path = 'path/to/checkpoint.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        energy_threshold = checkpoint.get('energy_threshold', None)
    else:
        energy_threshold = None
    
    # Load test data
    _, _, test_loader = get_rml_dataloaders(
        data_path='/path/to/data',
        experiment_setting=experiment_setting,
        batch_size=64,
        num_workers=4
    )
    
    # Note: You'll need to modify the test_loader or dataset to include 
    # original modulation information for unknown samples
    
    # Run evaluation
    metrics = evaluator.evaluate_per_unknown_class(
        model=model,
        test_loader=test_loader,
        device=device,
        energy_threshold=energy_threshold
    )
    
    # Print results
    print("\n" + "="*60)
    print("PER-UNKNOWN-CLASS OOD DETECTION RESULTS")
    print("="*60)
    
    for key, value in sorted(metrics.items()):
        if isinstance(value, (list, np.ndarray)):
            print(f"{key}: {value}")
        elif isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    return metrics


if __name__ == "__main__":
    import os
    demo_per_unknown_evaluation()
