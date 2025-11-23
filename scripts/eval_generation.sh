#!/bin/bash

# Generative Quality Evaluation Script for MeanFlow
# Evaluates the generative ability of the trained model on RML2016.10a dataset

set -e  # Exit on error

# Default values
CHECKPOINT="meanflow/best_model.pth"
DATA_PATH="data/RML2016.10a_dict.pkl"
SAMPLES_PER_CLASS=1000
BATCH_SIZE=256
METRICS="mmd,c2st,fid,psd,hist"
SNR_LEVELS="-10,0,10,18"  # Low, Medium, Medium, High SNR
CREATE_PLOTS=true
DEVICE=""  # Auto-detect if empty
EXPERIMENT_SETTING=""  # Use checkpoint's setting if empty
OUTPUT_DIR=""  # Auto-generate timestamp if empty

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --samples_per_class)
            SAMPLES_PER_CLASS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --metrics)
            METRICS="$2"
            shift 2
            ;;
        --snr_levels)
            SNR_LEVELS="$2"
            shift 2
            ;;
        --no-plots)
            CREATE_PLOTS=false
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --experiment_setting)
            EXPERIMENT_SETTING="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --checkpoint PATH          Path to checkpoint file (default: meanflow/best_model.pth)"
            echo "  --data_path PATH           Path to RML2016.10a dataset (default: data/RML2016.10a_dict.pkl)"
            echo "  --samples_per_class N      Number of samples per class per SNR (default: 1000)"
            echo "  --batch_size N             Batch size for generation (default: 256)"
            echo "  --metrics LIST             Comma-separated metrics: mmd,c2st,fid,psd,hist (default: all)"
            echo "  --snr_levels LIST          Comma-separated SNR levels in dB (default: -10,0,10,18)"
            echo "  --no-plots                 Skip creating visualization plots"
            echo "  --device DEVICE            Device: cuda or cpu (auto-detect if not specified)"
            echo "  --experiment_setting N     Experiment setting 1-12 (uses checkpoint's if not specified)"
            echo "  --output_dir PATH          Output directory (auto-generated if not specified)"
            echo "  --help                     Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Quick evaluation (minimal metrics, no plots)"
            echo "  $0 --samples_per_class 100 --metrics mmd,c2st --no-plots"
            echo ""
            echo "  # Full evaluation with custom checkpoint"
            echo "  $0 --checkpoint outputs/modulation/best_model.pth --samples_per_class 2000"
            echo ""
            echo "  # Evaluate at specific SNR levels"
            echo "  $0 --snr_levels \"-10,0,10,18\""
            echo ""
            echo "  # CPU-only evaluation"
            echo "  $0 --device cpu"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT"
    echo "Please provide a valid checkpoint path with --checkpoint"
    exit 1
fi

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Dataset file not found: $DATA_PATH"
    echo "Please provide a valid dataset path with --data_path"
    exit 1
fi

# Build command
CMD="python -m meanflow.scripts.generative_eval_rml"
CMD="$CMD --checkpoint $CHECKPOINT"
CMD="$CMD --data_path $DATA_PATH"
CMD="$CMD --samples_per_class $SAMPLES_PER_CLASS"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --metrics $METRICS"
CMD="$CMD --snr_levels $SNR_LEVELS"

if [ "$CREATE_PLOTS" = true ]; then
    CMD="$CMD --plots"
fi

if [ -n "$DEVICE" ]; then
    CMD="$CMD --device $DEVICE"
fi

if [ -n "$EXPERIMENT_SETTING" ]; then
    CMD="$CMD --experiment_setting $EXPERIMENT_SETTING"
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
fi

# Print configuration
echo "=========================================="
echo "Generative Quality Evaluation"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Data path: $DATA_PATH"
echo "Samples per class per SNR: $SAMPLES_PER_CLASS"
echo "Batch size: $BATCH_SIZE"
echo "Metrics: $METRICS"
echo "SNR levels: $SNR_LEVELS"
echo "Create plots: $CREATE_PLOTS"
if [ -n "$DEVICE" ]; then
    echo "Device: $DEVICE"
else
    echo "Device: auto-detect"
fi
if [ -n "$EXPERIMENT_SETTING" ]; then
    echo "Experiment setting: $EXPERIMENT_SETTING"
else
    echo "Experiment setting: from checkpoint"
fi
echo "=========================================="
echo ""

# Run evaluation
echo "Starting evaluation..."
$CMD

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "=========================================="
echo "Check outputs/generative_eval/ for results"
echo "  - metrics.json: All computed metrics (per-class, per-SNR, and SNR categories)"
echo "  - plots/: Visualization plots for each class at key SNR levels"
echo ""
echo "SNR Category Summary:"
echo "  - Low SNR: < -5 dB (challenging conditions)"
echo "  - Medium SNR: -5 to 15 dB (typical conditions)"
echo "  - High SNR: > 15 dB (ideal conditions)"


