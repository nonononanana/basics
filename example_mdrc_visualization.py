#!/usr/bin/env python3
"""
Example: How to use MDRC visualization feature

This script demonstrates how to enable visualization when evaluating
OOD detection with the MDRC method.
"""

import subprocess
import sys
from pathlib import Path

def run_mdrc_with_visualization(
    checkpoint_path: str,
    data_path: str,
    experiment_setting: int = 1,
    output_dir: str = "./mdrc_results"
):
    """
    Run MDRC evaluation with visualization enabled.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        data_path: Path to RML dataset
        experiment_setting: Experiment setting (1-12)
        output_dir: Directory to save results and visualizations
    """
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, "-m", "meanflow.evaluate_ood_denoising",
        "--checkpoint", checkpoint_path,
        "--data_path", data_path,
        "--experiment_setting", str(experiment_setting),
        "--method", "mdrc",
        "--save_visualization",  # Enable visualization
        "--output_dir", output_dir,
        "--batch_size", "512",
        "--snr_min", "-20",
        "--snr_max", "20"
    ]
    
    print("Running MDRC evaluation with visualization...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run evaluation
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print("\n✓ Evaluation completed successfully!")
        print(f"\nVisualization images saved to: {output_dir}/eval_mdrc/")
        print("\nGenerated files:")
        print("  - class_00_noisy.png     (noisy signal for class 0)")
        print("  - class_00_denoised.png  (denoised signal for class 0)")
        print("  - class_00_residual.png  (residual for class 0)")
        print("  - ... (repeat for classes 1-10)")
        print("\nTotal: 33 images (11 classes × 3 images per class)")
    else:
        print("\n✗ Evaluation failed!")
        sys.exit(1)


def run_mdrc_without_visualization(
    checkpoint_path: str,
    data_path: str,
    experiment_setting: int = 1
):
    """
    Run MDRC evaluation without visualization (faster).
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        data_path: Path to RML dataset
        experiment_setting: Experiment setting (1-12)
    """
    
    # Build command (no --save_visualization flag)
    cmd = [
        sys.executable, "-m", "meanflow.evaluate_ood_denoising",
        "--checkpoint", checkpoint_path,
        "--data_path", data_path,
        "--experiment_setting", str(experiment_setting),
        "--method", "mdrc",
        "--batch_size", "512",
        "--snr_min", "-20",
        "--snr_max", "20"
    ]
    
    print("Running MDRC evaluation without visualization...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run evaluation
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print("\n✓ Evaluation completed successfully!")
        print("No visualization images saved (use --save_visualization to enable).")
    else:
        print("\n✗ Evaluation failed!")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 70)
    print("MDRC Visualization Example")
    print("=" * 70)
    print()
    print("This script shows how to use the MDRC visualization feature.")
    print()
    print("Usage:")
    print()
    print("  # With visualization (generates 33 images)")
    print("  python example_mdrc_visualization.py --with-viz \\")
    print("      --checkpoint path/to/checkpoint.pt \\")
    print("      --data_path path/to/data")
    print()
    print("  # Without visualization (faster)")
    print("  python example_mdrc_visualization.py --no-viz \\")
    print("      --checkpoint path/to/checkpoint.pt \\")
    print("      --data_path path/to/data")
    print()
    print("=" * 70)
    print()
    
    # Parse simple command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='MDRC Visualization Example')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--experiment_setting', type=int, default=1,
                       help='Experiment setting (1-12)')
    parser.add_argument('--output_dir', type=str, default='./mdrc_results',
                       help='Output directory for visualization')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--with-viz', action='store_true',
                      help='Enable visualization (generates 33 images)')
    group.add_argument('--no-viz', action='store_true',
                      help='Disable visualization (faster)')
    
    args = parser.parse_args()
    
    if args.with_viz:
        run_mdrc_with_visualization(
            checkpoint_path=args.checkpoint,
            data_path=args.data_path,
            experiment_setting=args.experiment_setting,
            output_dir=args.output_dir
        )
    else:
        run_mdrc_without_visualization(
            checkpoint_path=args.checkpoint,
            data_path=args.data_path,
            experiment_setting=args.experiment_setting
        )

