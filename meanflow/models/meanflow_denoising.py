"""
Mean Flow Model for Signal Denoising
Implements conditional denoising with modulation type as condition
Learns to map noisy signals to clean signals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
from meanflow.models.ema import init_ema, update_ema_net
import logging

logger = logging.getLogger(__name__)


class MeanFlowDenoising(nn.Module):
    """
    Mean Flow model adapted for signal denoising
    Uses modulation type as condition for denoising
    Input: noisy signal, Output: clean signal
    """
    
    def __init__(
        self, 
        arch,  # UNet architecture
        args,  # Training arguments
        net_configs,  # Network configuration
        num_classes: int = 9,  # Number of known modulation classes
        class_embed_dim: int = 128  # Dimension of class embedding
    ):
        """
        Initialize Mean Flow model for denoising
        
        Args:
            arch: Neural network architecture (e.g., UNet)
            args: Training arguments
            net_configs: Network configuration dict
            num_classes: Number of known modulation classes
            class_embed_dim: Dimension of class embedding
        """
        super(MeanFlowDenoising, self).__init__()
        
        self.num_classes = num_classes
        self.class_embed_dim = class_embed_dim
        self.args = args
        
        # Main network with class conditioning
        self.net = arch(**net_configs)
        
        # Class embedding layer for conditioning
        self.class_embedding = nn.Embedding(
            num_embeddings=num_classes,
            embedding_dim=class_embed_dim
        )
        
        # Initialize embeddings with small random values
        nn.init.normal_(self.class_embedding.weight, mean=0.0, std=0.02)
        
        # Buffer for tracking updates (for EMA)
        self.register_buffer("num_updates", torch.tensor(0))
        
        # EMA networks for stable generation
        self.net_ema = init_ema(self.net, arch(**net_configs), args.ema_decay)
        
        # Additional EMA networks with different decay rates
        self.ema_decays = args.ema_decays
        for i, ema_decay in enumerate(self.ema_decays):
            self.add_module(
                f"net_ema{i + 1}", 
                init_ema(self.net, arch(**net_configs), ema_decay)
            )
    
    def update_ema(self):
        """Update EMA networks"""
        self.num_updates += 1
        num_updates = self.num_updates
        
        # Update main EMA network
        update_ema_net(self.net, self.net_ema, num_updates)
        
        # Update additional EMA networks
        for i in range(len(self.ema_decays)):
            update_ema_net(self.net, self._modules[f"net_ema{i + 1}"], num_updates)
    
    def get_class_embeddings(self, class_labels: torch.Tensor) -> torch.Tensor:
        """
        Get class embeddings for given labels
        
        Args:
            class_labels: Tensor of class labels [batch_size]
        
        Returns:
            Class embeddings [batch_size, class_embed_dim]
        """
        # Handle -1 labels (unknown classes) by mapping to zero embedding
        valid_mask = class_labels >= 0
        embeddings = torch.zeros(
            class_labels.size(0), 
            self.class_embed_dim, 
            device=class_labels.device
        )
        
        if valid_mask.any():
            valid_labels = class_labels[valid_mask]
            embeddings[valid_mask] = self.class_embedding(valid_labels)
        
        return embeddings
    
    def forward_with_loss(
        self, 
        x_noisy: torch.Tensor,
        x_clean: torch.Tensor, 
        class_labels: torch.Tensor,
        aug_cond: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with loss computation for denoising task
        
        Learns to predict x_clean from x_noisy.
        
        Args:
            x_noisy: Noisy signal data [batch_size, 2, 128] (I/Q channels)
            x_clean: Clean signal data [batch_size, 2, 128] (I/Q channels)
            class_labels: Modulation class labels [batch_size]
            aug_cond: Optional augmentation conditioning
        
        Returns:
            Dictionary containing loss components
        """
        device = x_noisy.device
        batch_size = x_noisy.shape[0]
        
        # Sample timestep t uniformly from [0, 1]
        t = torch.rand(batch_size, device=device)
        
        # Reshape for broadcasting
        t_reshaped = t.view(-1, 1, 1)
        
        # Linear interpolation between clean (t=0) and noisy (t=1)
        z = (1 - t_reshaped) * x_clean + t_reshaped * x_noisy
        
        # Predict clean signal directly
        # We pass t for time embedding. 
        # Note: We now pass single t instead of (t, t) tuple as we simplify UNet interface
        x_pred = self.net(
            z, 
            t, 
            aug_cond,
            class_labels=class_labels,
            noisy_cond=None
        )
        
        # Simple MSE loss on x prediction
        loss = F.mse_loss(x_pred, x_clean)
        
        return {
            'total_loss': loss,
            'loss': loss
        }
    
    def denoise(
        self, 
        x_noisy: torch.Tensor,
        class_labels: torch.Tensor,
        net: Optional[nn.Module] = None,
        num_steps: int = 1
    ) -> torch.Tensor:
        """
        Denoise a noisy signal conditioned on modulation type
        
        Args:
            x_noisy: Noisy input signal [batch_size, 2, 128]
            class_labels: Class labels for conditional denoising
            net: Network to use (default: self.net_ema)
            num_steps: Number of denoising steps (1 for single-step)
        
        Returns:
            Denoised signal [batch_size, 2, 128]
        """
        net = net if net is not None else self.net_ema
        batch_size = x_noisy.shape[0]
        device = x_noisy.device
        
        if num_steps == 1:
            # Single-step prediction: directly predict x_clean from x_noisy (t=1)
            z = x_noisy
            t = torch.ones(batch_size, device=device)
            
            x_pred = net(z, t, aug_cond=None, class_labels=class_labels, noisy_cond=None)
            x_clean = x_pred
        else:
            # Multi-step sampling using Euler method on the ODE
            # ODE: dz/dt = v = (z - x) / t
            z = x_noisy.clone()
            dt = 1.0 / num_steps
            
            for step in range(num_steps):
                t_val = 1.0 - step * dt
                t = torch.full((batch_size,), t_val, device=device)
                
                # Predict x_clean at current step
                x_pred = net(z, t, aug_cond=None, class_labels=class_labels, noisy_cond=None)
                
                # Compute velocity: v = (z - x) / t
                # Avoid division by zero at t=0
                if t_val > 1e-5:
                    v = (z - x_pred) / t_val
                    # Euler update: z_{t-dt} = z_t - v * dt
                    z = z - v * dt
                else:
                    z = x_pred

            x_clean = z
        
        return x_clean
    
    def sample(
        self, 
        x_noisy: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        net: Optional[nn.Module] = None,
        num_steps: int = 1
    ) -> torch.Tensor:
        """
        Sample from the model (denoise signals)
        
        This is an alias for the denoise method to maintain API consistency
        
        Args:
            x_noisy: Noisy input signal [batch_size, 2, 128]
            class_labels: Class labels for conditional generation
            net: Network to use (default: self.net_ema)
            num_steps: Number of denoising steps
        
        Returns:
            Denoised samples
        """
        return self.denoise(x_noisy, class_labels, net, num_steps)

