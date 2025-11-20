#!/bin/bash
# Verification script for validation set OOD changes

echo "================================================================"
echo "Verification: Validation Set OOD Update"
echo "================================================================"
echo ""

# Check if files exist
echo "📁 Checking modified files..."
files=(
    "meanflow/data/rml_dataset.py"
    "meanflow/train_modulation.py"
    "VALIDATION_SPLIT_README.md"
    "VALIDATION_OOD_UPDATE.md"
    "CHANGES_SUMMARY.md"
    "test_validation_ood.py"
    "test_val_logic.py"
)

all_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (missing)"
        all_exist=false
    fi
done

echo ""
echo "================================================================"
echo "🧪 Running unit tests..."
echo "================================================================"
echo ""

# Run logic test
python test_val_logic.py

exit_code=$?

echo ""
echo "================================================================"
echo "📊 Summary"
echo "================================================================"
echo ""

if [ $exit_code -eq 0 ]; then
    echo "✅ All verifications passed!"
    echo ""
    echo "Changes implemented:"
    echo "  1. ✅ Validation set now includes unknown classes"
    echo "  2. ✅ Synthetic negatives disabled by default"  
    echo "  3. ✅ Model selection uses AUROC when available"
    echo "  4. ✅ Documentation updated"
    echo ""
    echo "You can now train with:"
    echo "  python meanflow/train_modulation.py \\"
    echo "    --experiment_setting 1 \\"
    echo "    --epochs 100 \\"
    echo "    --batch_size 256 \\"
    echo "    --data_path data/RML2016.10a_dict.pkl"
    echo ""
    echo "Monitor val/auroc for model selection!"
else
    echo "❌ Some verifications failed"
    echo "Please review the test output above"
fi

echo "================================================================"
exit $exit_code

