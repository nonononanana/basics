#!/bin/bash

# Simple OOD detection using reconstruction error on negative samples
# This approach should actually work and fix the 0% accuracies

echo "Testing simple reconstruction-based OOD detection..."

cd /Users/Axer/Desktop/py-meanflow

# Run training with simplified approach:
# - No complex energy losses 
# - Use reconstruction error for OOD detection
# - Train on positive samples for reconstruction
# - Use negative samples to learn what "doesn't belong"

python -m meanflow.train_modulation \
    --data_path data/RML2016.10a_dict.pkl \
    --experiment_setting 1 \
    --epochs 30 \
    --eval_freq 3 \
    --batch_size 256 \
    --lr 2e-4 \
    --lambda_rec 1.0 \
    --lambda_arc 0.3 \
    --lambda_ood 0.5 \
    --use_arcface \
    --arcface_margin 0.3 \
    --arcface_scale 15.0 \
    --target_fpr 0.05 \
    --output_dir outputs/simple_ood \
    --wandb_name simple_ood_exp1 \
    --device cuda \
    --seed 42

echo ""
echo "Simple approach implemented:"
echo "1. Removed complex energy-based losses"
echo "2. Use reconstruction error for classification and OOD detection"
echo "3. Train model to reconstruct known classes well"
echo "4. Unknown samples should have higher reconstruction error"
echo "5. Threshold based on reconstruction error percentiles"
echo ""
echo "This should fix the 0% accuracies by using a more straightforward approach."
