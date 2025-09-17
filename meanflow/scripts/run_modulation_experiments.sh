#!/bin/bash

# Script to run modulation classification experiments
# This script runs multiple experimental settings with different configurations

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Base configuration
DATA_PATH="data/RML2016.10a_dict.pkl"
OUTPUT_BASE="outputs/modulation"
WANDB_PROJECT="meanflow-modulation"

# Function to run single experiment
run_experiment() {
    local exp_setting=$1
    local use_arcface=$2
    local model_channels=$3
    local dropout=$4
    local weight_decay=$5
    local exp_name=$6
    
    echo "========================================"
    echo "Running Experiment: ${exp_name}"
    echo "Setting: ${exp_setting}, ArcFace: ${use_arcface}"
    echo "========================================"
    
    # Create output directory
    output_dir="${OUTPUT_BASE}/${exp_name}"
    mkdir -p ${output_dir}
    
    # Build command
    cmd="python meanflow/train_modulation.py"
    cmd="${cmd} --data_path ${DATA_PATH}"
    cmd="${cmd} --experiment_setting ${exp_setting}"
    cmd="${cmd} --output_dir ${output_dir}"
    cmd="${cmd} --epochs 100"
    cmd="${cmd} --batch_size 64"
    cmd="${cmd} --eval_batch_size 128"
    cmd="${cmd} --lr 1e-4"
    cmd="${cmd} --warmup_epochs 5"
    cmd="${cmd} --model_channels ${model_channels}"
    cmd="${cmd} --num_blocks 2"
    cmd="${cmd} --dropout ${dropout}"
    cmd="${cmd} --weight_decay ${weight_decay}"
    cmd="${cmd} --mixed_precision"
    cmd="${cmd} --grad_clip 1.0"
    cmd="${cmd} --eval_freq 5"
    cmd="${cmd} --num_workers 4"
    cmd="${cmd} --wandb_project ${WANDB_PROJECT}"
    cmd="${cmd} --wandb_name ${exp_name}"
    
    # Add ArcFace if enabled
    if [ "${use_arcface}" = "true" ]; then
        cmd="${cmd} --use_arcface"
        cmd="${cmd} --arcface_margin 0.5"
        cmd="${cmd} --arcface_scale 30.0"
    fi
    
    # Log command
    echo "Command: ${cmd}"
    echo ""
    
    # Run experiment
    ${cmd} 2>&1 | tee "${output_dir}/train.log"
    
    echo "Experiment ${exp_name} completed!"
    echo ""
}

# Experiment 1: Baseline without ArcFace (Setting 1: 9 known, 2 unknown)
run_experiment 1 false 64 0.1 1e-4 "exp1_baseline"

# Experiment 2: With ArcFace (Setting 1: 9 known, 2 unknown)
run_experiment 1 true 64 0.1 1e-4 "exp1_arcface"

# Experiment 3: Increased regularization (Setting 1)
run_experiment 1 true 64 0.2 5e-4 "exp1_high_reg"

# Experiment 4: Smaller model (Setting 1)
run_experiment 1 true 32 0.15 1e-4 "exp1_small_model"

# Experiment 5: Balanced split (Setting 2: 6 known, 5 unknown)
run_experiment 2 true 64 0.1 1e-4 "exp2_arcface"

# Experiment 6: Extreme unknown (Setting 4: 3 known, 8 unknown)
run_experiment 4 true 64 0.15 2e-4 "exp4_arcface"

# Experiment 7: Many known classes (Setting 9: 9 known, 2 unknown from 04C)
run_experiment 9 true 64 0.1 1e-4 "exp9_arcface"

# Experiment 8: Challenging split (Setting 11: 5 known, 6 unknown)
run_experiment 11 true 64 0.15 2e-4 "exp11_arcface"

echo "========================================"
echo "All experiments completed!"
echo "Results saved to: ${OUTPUT_BASE}"
echo "========================================"
