# Quick Reference: Generative Evaluation

## Fastest Way to Run

```bash
bash scripts/eval_generation.sh
```

## Custom Run

```bash
bash scripts/eval_generation.sh \
  --checkpoint meanflow/best_model.pth \
  --samples_per_class 1000 \
  --batch_size 256 \
  --metrics mmd,c2st,fid,psd,hist \
  --device cuda
```

## Quick Test (Fast, Minimal)

```bash
bash scripts/eval_generation.sh \
  --samples_per_class 100 \
  --metrics mmd,c2st \
  --no-plots \
  --batch_size 128
```

## Output Location

Results are saved to: `outputs/generative_eval/TIMESTAMP/`

- `metrics.json` - All metrics in JSON format
- `plots/` - Visualization plots for each class

## Key Metrics to Check

- **MMD**: Should be < 0.01 for good quality
- **C2ST**: Should be ~50-55% (lower = better, means indistinguishable)
- **FID**: Should be < 1.0 for good semantic match
- **PSD Distance**: Should be < 0.1 for spectral match

## Help

```bash
bash scripts/eval_generation.sh --help
```

For detailed documentation, see `GENERATION_EVAL.md`


# Generative Quality Evaluation Guide

This guide helps you evaluate the generative ability of your trained MeanFlow model on the RML2016.10a dataset.

## Quick Start

### Prerequisites

1. **Dataset**: `data/RML2016.10a_dict.pkl` (should already be in your workspace)
2. **Checkpoint**: `meanflow/best_model.pth` (your trained model checkpoint)
3. **Environment**: Activate your conda environment:
   ```bash
   conda activate meanflow  # or your environment name
   ```

### One-Command Evaluation

Run the evaluation script using the provided bash script:

```bash
bash scripts/eval_generation.sh
```

Or run directly with Python:

```bash
python -m meanflow.scripts.generative_eval_rml \
  --checkpoint meanflow/best_model.pth \
  --data_path data/RML2016.10a_dict.pkl \
  --samples_per_class 1000 \
  --batch_size 256 \
  --metrics mmd,c2st,fid,psd,hist \
  --plots
```

## Understanding the Metrics

The evaluation script computes several metrics to assess generative quality:

### 1. **MMD (Maximum Mean Discrepancy)**
- **What it measures**: Statistical distance between real and generated distributions
- **Interpretation**: Lower is better. Values close to 0 indicate good distribution match
- **Range**: [0, ∞)

### 2. **C2ST (Classifier Two-Sample Test) Accuracy**
- **What it measures**: How well a classifier can distinguish real from generated samples
- **Interpretation**: 
  - ~50% = samples are indistinguishable (good!)
  - >50% = classifier can tell them apart (higher = more distinguishable)
- **Range**: [0.5, 1.0]

### 3. **FID-like (Fréchet Distance in Feature Space)**
- **What it measures**: Distance between real and generated samples in the model's learned feature space
- **Interpretation**: Lower is better. Measures semantic similarity
- **Range**: [0, ∞)

### 4. **PSD Distance (Power Spectral Density)**
- **What it measures**: L2 distance between mean power spectral densities
- **Interpretation**: Lower is better. Indicates spectral match between signals
- **Range**: [0, ∞)

### 5. **Histogram Distances (Amplitude & Phase JS)**
- **What it measures**: Jensen-Shannon divergence between amplitude/phase distributions
- **Interpretation**: Lower is better. Values close to 0 indicate distributional match
- **Range**: [0, 1] for JS divergence

## Command-Line Options

```bash
python -m meanflow.scripts.generative_eval_rml \
  --checkpoint PATH_TO_CHECKPOINT.pth \      # Required: path to model checkpoint
  --data_path data/RML2016.10a_dict.pkl \    # Path to dataset (default: data/RML2016.10a_dict.pkl)
  --experiment_setting AUTO \                 # Experiment setting (1-12) or AUTO to use checkpoint's setting
  --device cuda \                             # Device: cuda or cpu (auto-detects if not specified)
  --samples_per_class 1000 \                  # Number of samples to generate per class
  --batch_size 256 \                          # Batch size for generation
  --metrics mmd,c2st,fid,psd,hist \          # Comma-separated metrics to compute
  --plots \                                   # Create visualization plots
  --output_dir outputs/generative_eval/run1  # Output directory (default: auto-generated timestamp)
```

### Available Metrics

You can select any subset of metrics:
- `mmd` - Maximum Mean Discrepancy
- `c2st` - Classifier Two-Sample Test
- `fid` - Fréchet Distance in feature space
- `psd` - Power Spectral Density distance
- `hist` - Histogram distances (amplitude & phase)

Example: `--metrics mmd,c2st` to compute only MMD and C2ST.

## Output Structure

After running the evaluation, you'll find:

```
outputs/generative_eval/TIMESTAMP/
├── metrics.json              # All computed metrics (JSON format)
└── plots/
    ├── 8PSK_visualization.png
    ├── AM-DSB_visualization.png
    ├── BPSK_visualization.png
    └── ... (one plot per known class)
```

### metrics.json Structure

```json
{
  "experiment_setting": 1,
  "known_classes": ["8PSK", "AM-DSB", "BPSK", ...],
  "num_samples_per_class": 1000,
  "per_class_metrics": {
    "8PSK": {
      "mmd": 0.001234,
      "c2st_accuracy": 0.52,
      "fid": 0.567890,
      "psd_distance": 0.123456,
      "amplitude_js": 0.012345,
      "phase_js": 0.023456
    },
    ...
  },
  "global_metrics": {
    "mmd": 0.002345,
    "c2st_accuracy": 0.53,
    ...
  }
}
```

## Visualization Plots

Each class visualization includes 4 subplots:

1. **Constellation Plot**: I/Q scatter plot showing signal points in complex plane
2. **Amplitude Distribution**: Histogram comparing amplitude distributions
3. **Phase Distribution**: Histogram comparing phase distributions  
4. **Power Spectral Density**: Comparison of frequency domain characteristics

## Interpreting Results

### Good Generative Quality Indicators:
- **MMD < 0.01**: Strong distribution match
- **C2ST ≈ 50-55%**: Samples are nearly indistinguishable
- **FID < 1.0**: Good semantic similarity
- **PSD Distance < 0.1**: Spectral characteristics match well
- **JS Divergence < 0.1**: Amplitude/phase distributions match

### Warning Signs:
- **C2ST > 70%**: Generated samples are easily distinguishable from real
- **MMD > 0.1**: Significant distribution mismatch
- **Large PSD Distance**: Spectral characteristics don't match

## Troubleshooting

### Issue: "Checkpoint does not contain 'args'"
**Solution**: Make sure you're using a checkpoint saved by `train_modulation.py`. The checkpoint must include the training arguments.

### Issue: "Only N samples available for class X"
**Solution**: Reduce `--samples_per_class` to match available test samples, or use a different split.

### Issue: Out of memory errors
**Solution**: 
- Reduce `--batch_size` (e.g., `--batch_size 128`)
- Reduce `--samples_per_class` (e.g., `--samples_per_class 500`)
- Use CPU: `--device cpu` (slower but uses less memory)

### Issue: CUDA out of memory
**Solution**: The script automatically falls back to CPU if CUDA fails, but you can force CPU:
```bash
--device cpu
```

## Example Workflow

1. **Quick sanity check** (fast, minimal metrics):
   ```bash
   python -m meanflow.scripts.generative_eval_rml \
     --checkpoint meanflow/best_model.pth \
     --samples_per_class 100 \
     --metrics mmd,c2st \
     --batch_size 128
   ```

2. **Full evaluation** (comprehensive, takes longer):
   ```bash
   python -m meanflow.scripts.generative_eval_rml \
     --checkpoint meanflow/best_model.pth \
     --samples_per_class 1000 \
     --metrics mmd,c2st,fid,psd,hist \
     --plots \
     --batch_size 256
   ```

3. **Per-class analysis** (check specific classes):
   - Run full evaluation with `--plots`
   - Examine individual class visualizations in `plots/` directory
   - Check per-class metrics in `metrics.json`

## Advanced Usage

### Custom Experiment Setting

If you want to evaluate with a different experiment setting than the checkpoint:

```bash
python -m meanflow.scripts.generative_eval_rml \
  --checkpoint meanflow/best_model.pth \
  --experiment_setting 2 \
  --samples_per_class 1000
```

### SNR-Specific Evaluation

The script uses all SNR values in the test set by default. To evaluate specific SNR ranges, you would need to modify the dataset loading in the script (future enhancement).

## Next Steps

After evaluation:

1. **Review metrics.json**: Check if metrics indicate good generative quality
2. **Examine plots**: Visual inspection of constellation, amplitude, phase, and PSD
3. **Compare classes**: Some modulation types may be easier/harder to generate
4. **Iterate**: If quality is poor, consider:
   - Training longer
   - Adjusting hyperparameters
   - Checking training data quality

## Questions?

- Check the script help: `python -m meanflow.scripts.generative_eval_rml --help`
- Review the code: `meanflow/scripts/generative_eval_rml.py`
- Check training logs for any issues during model training


