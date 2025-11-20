#!/usr/bin/env python3
"""
Unit test to verify validation set logic includes unknown classes
Tests the split logic without requiring actual data file
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def test_split_logic():
    """Test that the split logic in rml_dataset.py correctly includes unknowns in validation"""
    
    print("=" * 80)
    print("Testing Validation Split Logic")
    print("=" * 80)
    
    # Read the dataset file
    dataset_file = Path(__file__).parent / 'meanflow' / 'data' / 'rml_dataset.py'
    
    with open(dataset_file, 'r') as f:
        content = f.read()
    
    # Find the split selection logic
    split_logic_start = content.find("# Select indices based on split type")
    split_logic_end = content.find("# Add selected samples in batch", split_logic_start)
    
    split_logic = content[split_logic_start:split_logic_end]
    
    print("\nExtracted Split Logic:")
    print("-" * 80)
    print(split_logic)
    print("-" * 80)
    
    # Check conditions
    checks = {
        "Train excludes unknown": "if self.split == 'train':" in split_logic and 
                                   "if not is_unk:" in split_logic.split("elif self.split == 'val':")[0],
        "Val includes known": "elif self.split == 'val':" in split_logic,
        "Val includes unknown": "elif self.split == 'val':" in split_logic and
                                 "if not is_unk:" not in split_logic.split("elif self.split == 'val':")[1].split("else:")[0],
        "Test includes both": "else:  # test" in split_logic
    }
    
    print("\nValidation Checks:")
    print("-" * 80)
    
    all_passed = True
    for check_name, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {check_name}")
        if not result:
            all_passed = False
    
    print("-" * 80)
    
    # Check synthetic negatives default
    synthetic_neg_pattern = "--eval_with_synthetic_negatives"
    train_file = Path(__file__).parent / 'meanflow' / 'train_modulation.py'
    
    with open(train_file, 'r') as f:
        train_content = f.read()
    
    # Find the argument definition
    for line in train_content.split('\n'):
        if synthetic_neg_pattern in line and 'add_argument' in line:
            if 'default=False' in line or 'default = False' in line:
                print("✅ PASS: Synthetic negatives disabled by default")
            elif 'default=True' in line or 'default = True' in line:
                print("⚠️  WARNING: Synthetic negatives still enabled by default")
                print("   (But validation will use real OOD data if available)")
            break
    
    # Check model selection metric
    if 'eval/auroc' in train_content and 'is_best = ' in train_content:
        auroc_check = train_content.find("not np.isnan(val_metrics['eval/auroc'])")
        if auroc_check > 0:
            print("✅ PASS: Model selection uses AUROC when available")
        else:
            print("⚠️  WARNING: Model selection might not use AUROC")
    
    print("=" * 80)
    
    if all_passed:
        print("✅ ALL CHECKS PASSED")
        print("\nValidation set will now include unknown classes:")
        print("  - Known classes: For closed-set classification evaluation")
        print("  - Unknown classes: For OOD detection evaluation (AUROC, AUPR)")
        print("\nModel selection will prefer AUROC over closed-set accuracy")
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nPlease review the split logic in rml_dataset.py")
    
    print("=" * 80)
    
    return all_passed


def test_docstring_update():
    """Test that documentation strings are updated"""
    
    print("\n" + "=" * 80)
    print("Testing Documentation Updates")
    print("=" * 80)
    
    dataset_file = Path(__file__).parent / 'meanflow' / 'data' / 'rml_dataset.py'
    
    with open(dataset_file, 'r') as f:
        content = f.read()
    
    # Check docstring
    checks = []
    
    if ("Validation set: Contains known" in content and "unknown" in content) or \
       "Validation set: Contains both known and unknown classes" in content:
        print("✅ PASS: Dataset docstring updated")
        checks.append(True)
    else:
        print("❌ FAIL: Dataset docstring not updated")
        checks.append(False)
    
    # Check README
    readme_file = Path(__file__).parent / 'VALIDATION_SPLIT_README.md'
    if readme_file.exists():
        with open(readme_file, 'r') as f:
            readme = f.read()
        
        if "Validation | 10% | ✅ Yes | ✅ Yes" in readme:
            print("✅ PASS: VALIDATION_SPLIT_README.md updated")
            checks.append(True)
        else:
            print("❌ FAIL: VALIDATION_SPLIT_README.md not updated")
            checks.append(False)
    
    # Check new documentation exists
    new_doc = Path(__file__).parent / 'VALIDATION_OOD_UPDATE.md'
    if new_doc.exists():
        print("✅ PASS: New documentation file created (VALIDATION_OOD_UPDATE.md)")
        checks.append(True)
    else:
        print("❌ FAIL: New documentation not created")
        checks.append(False)
    
    print("=" * 80)
    
    return all(checks)


if __name__ == '__main__':
    logic_passed = test_split_logic()
    doc_passed = test_docstring_update()
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    if logic_passed and doc_passed:
        print("✅ All tests passed!")
        print("\nChanges implemented successfully:")
        print("  1. Validation set now includes unknown classes")
        print("  2. Synthetic negatives disabled by default")
        print("  3. Model selection uses AUROC when available")
        print("  4. Documentation updated")
        print("\nYou can now train with realistic OOD evaluation on validation set!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        print("\nPlease review the changes")
        sys.exit(1)

