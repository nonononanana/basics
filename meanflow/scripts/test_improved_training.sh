#!/bin/bash

# Test script for improved training with proper loss weights and soft ranking loss
# This script tests the fixes for validation AUROC and energy separation

echo "Testing improved Mean Flow training with proper loss configuration..."

cd /Users/Axer/Desktop/py-meanflow

# Run training with:
# - Enabled negative energy loss (lambda_neg=0.2)
# - Soft ranking loss (lambda_rank=0.1)  
# - Proper margins for energy separation
# - Target FPR for validation threshold
# - Small ArcFace loss for class separation

python -m meanflow.train_modulation \
    --data_path data/RML2016.10a_dict.pkl \
    --experiment_setting 1 \
    --epochs 50 \
    --eval_freq 5 \
    --batch_size 256 \
    --lr 2e-4 \
    --lambda_rec 1.0 \
    --lambda_arc 0.2 \
    --lambda_pos 0.3 \
    --lambda_neg 0.2 \
    --lambda_rank 0.1 \
    --margin_pos -2.5 \
    --margin_neg -1.5 \
    --rank_margin 0.5 \
    --rank_beta 10.0 \
    --target_fpr 0.05 \
    --use_arcface \
    --arcface_margin 0.3 \
    --arcface_scale 15.0 \
    --output_dir outputs/improved_modulation \
    --wandb_name improved_exp1_proper_losses \
    --device cuda \
    --seed 42

echo "Training complete! Check outputs/improved_modulation for results."
echo ""
echo "Key improvements implemented:"
echo "1. Validation AUROC now handled correctly (NaN for val set without unknowns)"
echo "2. Threshold tuning uses target FPR (5%) on validation known samples"
echo "3. Soft ranking loss encourages energy separation between pos/neg samples"
echo "4. Checkpoint selection based on closed_set_accuracy instead of AUROC"
echo "5. Proper loss weights to learn energy boundaries"
echo ""
echo "Expected improvements:"
echo "- Test AUROC should increase significantly (>0.7)"
echo "- FPR should be around target (5%)"
echo "- Better separation between known/unknown energy distributions"
echo "- Improved closed-set accuracy with ArcFace"
