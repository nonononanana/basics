#!/bin/bash

# Test script for Mean Flow with ONLY reconstruction loss (original Mean Flow)
# All auxiliary losses disabled to train with pure Mean Flow objective

echo "Testing Mean Flow training with ONLY reconstruction loss..."

cd /userhome/MeanFlowOpenSetAMC

# Run training with:
# - ONLY reconstruction loss (lambda_rec=1.0)
# - All other losses disabled (lambda_arc/pos/neg/rank/cls/anchor=0.0)
# - Pure Mean Flow objective without energy-based or classification losses

python -m meanflow.train_modulation \
    --data_path data/RML2016.10a_dict.pkl \
    --experiment_setting 1 \
    --epochs 50 \
    --eval_freq 5 \
    --batch_size 256 \
    --lr 2e-4 \
    --lambda_rec 1.0 \
    --lambda_arc 0.0 \
    --lambda_pos 0.0 \
    --lambda_neg 0.0 \
    --lambda_rank 0.0 \
    --lambda_cls 0.0 \
    --lambda_anchor 0.0 \
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
echo "Configuration:"
echo "- ONLY reconstruction loss is used (lambda_rec=1.0)"
echo "- All other losses disabled (arc/pos/neg/rank/cls/anchor=0.0)"
echo "- This represents the original Mean Flow model without auxiliary losses"
echo ""
echo "Note:"
echo "- Training uses only the mean flow reconstruction objective"
echo "- No energy-based losses for OOD detection during training"
echo "- OOD detection at test time relies solely on reconstruction-based energy"
echo ""
echo "=========================================="
echo "Starting Generative Quality Evaluation..."
echo "=========================================="

# Run generative evaluation on the trained model
bash scripts/eval_generation.sh \
    --checkpoint outputs/improved_modulation/best_model.pth \
    --data_path data/RML2016.10a_dict.pkl \
    --experiment_setting 1 \
    --samples_per_class 1000 \
    --batch_size 256 \
    --metrics mmd,c2st,fid,psd,hist \
    --device cuda

echo ""
echo "=========================================="
echo "All tasks completed!"
echo "=========================================="
echo "Training results: outputs/improved_modulation/"
echo "Evaluation results: outputs/generative_eval/"

