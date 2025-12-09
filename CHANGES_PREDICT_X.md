# MeanFlow Denoising: Transition to x-prediction

## Overview
Based on the request to predict x (clean signal) instead of v (velocity) for the denoising task.

## Changes in `meanflow/models/meanflow_denoising.py`

### 1. `forward_with_loss`
- **Previous**: Learned velocity field $u$ matching target velocity $u_{tgt}$ derived from Mean Flow consistency.
  - Loss: $(u_{pred} - u_{tgt})^2$
- **New**: Directly predicts clean signal $x_{clean}$.
  - Loss: MSE($x_{pred}$, $x_{clean}$)
  - Network input: $z_t, t$ (same as before)
  - Network output interpretation: $x_{clean}$

### 2. `denoise`
- **Previous**:
  - Predicted $u$.
  - Update: $z_{new} = z - dt \cdot u$ (Euler step).
- **New**:
  - Predicts $x_{pred}$.
  - Single step: Returns $x_{pred}$.
  - Multi-step: Computes implied velocity $u = (z_t - x_{pred}) / t$ and performs Euler step.

## Rationale
Predicting $x$ directly simplifies the objective to a reconstruction loss, which is often stable and effective for denoising tasks, especially when the path is assumed to be linear (straight flow).

