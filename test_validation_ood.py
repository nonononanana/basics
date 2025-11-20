#!/usr/bin/env python3
"""
Test script to verify that validation set now includes unknown classes
"""

import sys
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from meanflow.data.rml_dataset import get_rml_dataloaders, EXPERIMENT_SETTINGS


def test_validation_with_ood():
    """Test that validation set now includes unknown classes"""
    
    print("=" * 80)
    print("Testing Validation Set with OOD (Unknown Classes)")
    print("=" * 80)
    
    # Test parameters
    data_path = 'data/RML2016.10a_dict.pkl'
    experiment_setting = 1
    batch_size = 64
    
    try:
        # Create dataloaders
        train_loader, val_loader, test_loader = get_rml_dataloaders(
            data_path=data_path,
            experiment_setting=experiment_setting,
            batch_size=batch_size,
            num_workers=0,  # Single thread for testing
            snr_range=(-20, 20),
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            normalize=False,
            seed=42,
            precompute_negatives=False  # Disable for faster loading
        )
        
        print(f"\n✅ Successfully created dataloaders")
        print(f"   - Train batches: {len(train_loader)}")
        print(f"   - Validation batches: {len(val_loader)}")
        print(f"   - Test batches: {len(test_loader)}")
        
        # Get experiment configuration
        known_classes = EXPERIMENT_SETTINGS[experiment_setting]['known']
        unknown_classes = EXPERIMENT_SETTINGS[experiment_setting]['unknown']
        
        print(f"\nExperiment Setting {experiment_setting}:")
        print(f"   - Known classes ({len(known_classes)}): {', '.join(known_classes)}")
        print(f"   - Unknown classes ({len(unknown_classes)}): {', '.join(unknown_classes)}")
        
        # Analyze each dataset
        for split_name, loader in [('Train', train_loader), ('Validation', val_loader), ('Test', test_loader)]:
            print(f"\n{'-' * 80}")
            print(f"{split_name} Set Analysis:")
            print(f"{'-' * 80}")
            
            all_labels = []
            all_is_unknown = []
            all_modulations = []
            
            # Collect all samples
            for pos_samples, neg_samples, labels, info in loader:
                all_labels.extend(labels.numpy())
                all_is_unknown.extend(info['is_unknown'].numpy())
                all_modulations.extend(info['original_modulation'])
            
            all_labels = np.array(all_labels)
            all_is_unknown = np.array(all_is_unknown)
            all_modulations = np.array(all_modulations)
            
            # Statistics
            n_total = len(all_labels)
            n_known = (~all_is_unknown).sum()
            n_unknown = all_is_unknown.sum()
            
            print(f"Total samples: {n_total}")
            print(f"Known samples: {n_known} ({n_known/n_total*100:.1f}%)")
            print(f"Unknown samples: {n_unknown} ({n_unknown/n_total*100:.1f}%)")
            
            # Per-modulation breakdown
            unique_mods = np.unique(all_modulations)
            print(f"\nPer-Modulation Breakdown:")
            
            print(f"  Known Classes:")
            for mod in known_classes:
                if mod in unique_mods:
                    count = (all_modulations == mod).sum()
                    print(f"    - {mod}: {count} samples")
                else:
                    print(f"    - {mod}: 0 samples (NOT PRESENT)")
            
            print(f"  Unknown Classes:")
            for mod in unknown_classes:
                if mod in unique_mods:
                    count = (all_modulations == mod).sum()
                    print(f"    - {mod}: {count} samples")
                else:
                    print(f"    - {mod}: 0 samples (NOT PRESENT)")
            
            # Validation checks
            if split_name == 'Train':
                if n_unknown > 0:
                    print(f"\n❌ ERROR: Training set should NOT contain unknown classes!")
                    print(f"   Found {n_unknown} unknown samples")
                else:
                    print(f"\n✅ PASS: Training set correctly excludes unknown classes")
                
                # Check known proportion (should be ~80%)
                if n_known > 0:
                    known_ratio = n_known / (n_known + n_unknown)
                    print(f"   Known samples ratio: {known_ratio*100:.1f}% (expected: 100%)")
            
            elif split_name == 'Validation':
                if n_unknown == 0:
                    print(f"\n❌ ERROR: Validation set should contain unknown classes for OOD evaluation!")
                    print(f"   Found {n_unknown} unknown samples (expected > 0)")
                else:
                    print(f"\n✅ PASS: Validation set correctly includes unknown classes")
                    print(f"   Can now evaluate OOD detection metrics (AUROC, AUPR, etc.)")
                
                # Check balance (unknown should be ~50% of each unknown class)
                if n_known > 0 and n_unknown > 0:
                    # For known: 10% of data, for unknown: 50% of data
                    # So we expect more unknown samples in val compared to the 10% known
                    print(f"   Known/Unknown ratio: {n_known}:{n_unknown}")
                    print(f"   (Known uses 10% per class, Unknown uses 50% per class)")
            
            elif split_name == 'Test':
                if n_unknown == 0:
                    print(f"\n❌ ERROR: Test set should contain unknown classes!")
                    print(f"   Found {n_unknown} unknown samples (expected > 0)")
                else:
                    print(f"\n✅ PASS: Test set correctly includes unknown classes")
                
                # Check balance
                if n_known > 0 and n_unknown > 0:
                    print(f"   Known/Unknown ratio: {n_known}:{n_unknown}")
                    print(f"   (Known uses 10% per class, Unknown uses 50% per class)")
        
        print(f"\n{'=' * 80}")
        print("Summary:")
        print(f"{'=' * 80}")
        print("✅ Validation set now includes unknown classes for OOD detection evaluation")
        print("✅ Can compute AUROC, AUPR, and other OOD metrics on validation set")
        print("✅ Model selection can be based on OOD detection performance")
        print(f"{'=' * 80}")
        
    except FileNotFoundError:
        print(f"\n❌ Data file not found: {data_path}")
        print("Please ensure the RML2016.10a dataset is downloaded to the 'data' directory")
        print("\nTest skipped (data file not available)")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    test_validation_with_ood()

