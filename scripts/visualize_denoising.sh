#!/bin/bash
# Quick script to visualize denoising and reconstruction results

# Default parameters
CHECKPOINT="${1:-outputs/denoising/best_model.pth}"
DATA_PATH="${2:-data/RML2016_denoising.pkl}"
EXPERIMENT_SETTING="${3:-1}"
MASK_RATIO="${4:-0.25}"
NUM_SAMPLES="${5:-10}"
OUTPUT_DIR="${6:-outputs/visualizations}"

echo "========================================="
echo "Denoising Visualization Script"
echo "========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Data Path: $DATA_PATH"
echo "Experiment Setting: $EXPERIMENT_SETTING"
echo "Mask Ratio: $MASK_RATIO"
echo "Number of Samples: $NUM_SAMPLES"
echo "Output Directory: $OUTPUT_DIR"
echo "========================================="
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint file not found: $CHECKPOINT"
    echo ""
    echo "Usage: $0 [CHECKPOINT] [DATA_PATH] [EXP_SETTING] [MASK_RATIO] [NUM_SAMPLES] [OUTPUT_DIR]"
    echo ""
    echo "Example:"
    echo "  $0 outputs/denoising/best_model.pth data/RML2016_denoising.pkl 1 0.25 10 outputs/viz"
    exit 1
fi

# Run visualization
python -m meanflow.visualize_denoising_reconstruction \
    --checkpoint "$CHECKPOINT" \
    --data_path "$DATA_PATH" \
    --experiment_setting "$EXPERIMENT_SETTING" \
    --mask_ratio "$MASK_RATIO" \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --dpi 150

echo ""
echo "========================================="
echo "Visualization complete!"
echo "Check results in: $OUTPUT_DIR"
echo "========================================="

