#!/bin/bash

# Script to run signal denoising experiments with Mean Flow
# Trains denoising model conditioned on modulation type

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Base configuration
DATA_PATH="data/RML2016_denoising.pkl"  # Path to 4-channel denoising dataset
OUTPUT_BASE="outputs/denoising"
WANDB_PROJECT="meanflow-denoising"

# Function to run single denoising experiment
run_denoising_experiment() {
    local exp_setting=$1
    local model_channels=$2
    local dropout=$3
    local lr=$4
    local epochs=$5
    local exp_name=$6
    
    echo "========================================"
    echo "Running Denoising Experiment: ${exp_name}"
    echo "Setting: ${exp_setting}, Channels: ${model_channels}"
    echo "========================================"
    
    # Create output directory
    output_dir="${OUTPUT_BASE}/${exp_name}"
    mkdir -p ${output_dir}
    
    # Build command
    cmd="python meanflow/train_denoising.py"
    cmd="${cmd} --data_path ${DATA_PATH}"
    cmd="${cmd} --experiment_setting ${exp_setting}"
    cmd="${cmd} --output_dir ${output_dir}"
    cmd="${cmd} --epochs ${epochs}"
    cmd="${cmd} --batch_size 256"
    cmd="${cmd} --eval_batch_size 512"
    cmd="${cmd} --lr ${lr}"
    cmd="${cmd} --warmup_epochs 5"
    cmd="${cmd} --model_channels ${model_channels}"
    cmd="${cmd} --num_blocks 2"
    cmd="${cmd} --dropout ${dropout}"
    cmd="${cmd} --weight_decay 1e-4"
    cmd="${cmd} --mixed_precision"
    cmd="${cmd} --grad_clip 1.0"
    cmd="${cmd} --eval_freq 5"
    cmd="${cmd} --num_workers 4"
    cmd="${cmd} --wandb_project ${WANDB_PROJECT}"
    cmd="${cmd} --wandb_name ${exp_name}"
    
    # Log command
    echo "Command: ${cmd}"
    echo ""
    
    # Run experiment
    ${cmd} 2>&1 | tee "${output_dir}/train.log"
    
    echo "Experiment ${exp_name} completed!"
    echo ""
}

# Function to run evaluation only
run_evaluation() {
    local exp_setting=$1
    local checkpoint_path=$2
    local exp_name=$3
    
    echo "========================================"
    echo "Running Evaluation: ${exp_name}"
    echo "========================================"
    
    output_dir="${OUTPUT_BASE}/${exp_name}_eval"
    mkdir -p ${output_dir}
    
    cmd="python meanflow/train_denoising.py"
    cmd="${cmd} --data_path ${DATA_PATH}"
    cmd="${cmd} --experiment_setting ${exp_setting}"
    cmd="${cmd} --output_dir ${output_dir}"
    cmd="${cmd} --resume ${checkpoint_path}"
    cmd="${cmd} --epochs 0"  # Evaluation only
    cmd="${cmd} --eval_batch_size 512"
    cmd="${cmd} --no_wandb"
    
    ${cmd} 2>&1 | tee "${output_dir}/eval.log"
}

# ============================================
# Main Experiments
# ============================================

echo "Starting Denoising Experiments..."
echo ""

# Experiment 1: Baseline configuration (Setting 1: 9 known, 2 unknown)
run_denoising_experiment 1 64 0.1 2e-4 100 "denoising_exp1_baseline"

# Experiment 2: Larger model (Setting 1)
run_denoising_experiment 1 128 0.1 1e-4 100 "denoising_exp1_large"

# Experiment 3: Higher dropout (Setting 1)
run_denoising_experiment 1 64 0.2 2e-4 100 "denoising_exp1_high_dropout"

# Experiment 4: Different modulation split (Setting 2: 6 known, 5 unknown)
run_denoising_experiment 2 64 0.1 2e-4 100 "denoising_exp2_baseline"

# Experiment 5: Extreme split (Setting 4: 3 known, 8 unknown)
run_denoising_experiment 4 64 0.15 2e-4 100 "denoising_exp4_baseline"

# Experiment 6: Many known classes (Setting 9: 9 known, 2 unknown)
run_denoising_experiment 9 64 0.1 2e-4 100 "denoising_exp9_baseline"

echo "========================================"
echo "All denoising experiments completed!"
echo "Results saved to: ${OUTPUT_BASE}"
echo "========================================"

