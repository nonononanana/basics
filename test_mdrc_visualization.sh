#!/bin/bash

# Test MDRC visualization
# This script demonstrates how to use the new visualization feature

# Example usage:
# python -m meanflow.evaluate_ood_denoising \
#     --checkpoint path/to/checkpoint.pt \
#     --data_path path/to/data \
#     --experiment_setting 1 \
#     --method mdrc \
#     --save_visualization \
#     --output_dir ./results

echo "MDRC Visualization Test Script"
echo "=============================="
echo ""
echo "To enable visualization when running OOD evaluation with MDRC method:"
echo ""
echo "  python -m meanflow.evaluate_ood_denoising \\"
echo "      --checkpoint YOUR_CHECKPOINT.pt \\"
echo "      --data_path YOUR_DATA_PATH \\"
echo "      --experiment_setting 1 \\"
echo "      --method mdrc \\"
echo "      --save_visualization \\"
echo "      --output_dir ./results"
echo ""
echo "This will create a 'eval_mdrc' folder in the output directory with:"
echo "  - 11 residual images (one per class)"
echo "  - 11 noisy signal images (before denoising)"
echo "  - 11 denoised signal images (after denoising)"
echo "  Total: 33 images"
echo ""
echo "Without --save_visualization flag, no images will be saved."



