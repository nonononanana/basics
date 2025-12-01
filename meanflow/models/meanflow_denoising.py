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

from meanflow.models.time_sampler import sample_two_timesteps
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
        
        The mean flow learns to predict the velocity field that transforms
        noisy signals to clean signals, conditioned on modulation type.
        
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
        
        # Sample noise for diffusion process
        e = torch.randn_like(x_clean).to(device)
        
        # Sample two timesteps for mean flow
        t, r = sample_two_timesteps(self.args, num_samples=batch_size, device=device)
        # Reshape for 1D signal broadcasting [batch, 1, 1]
        t = t.view(-1, 1, 1)
        r = r.view(-1, 1, 1)
        
        # Interpolate between clean data and noise
        # z_t = (1-t) * x_clean + t * e
        z = (1 - t) * x_clean + t * e
        v = e - x_clean  # Velocity field target
        
        # Define network function with class conditioning
        # The network takes noisy signal as additional input for conditioning
        def u_func(z, t, r):
            h = t - r
            # Pass class labels for modulation conditioning
            # Also pass noisy signal as condition to guide denoising
            return self.net(
                z, 
                (t.view(-1), h.view(-1)), 
                aug_cond,
                class_labels=class_labels,
                noisy_cond=x_noisy  # Condition on noisy signal
            )
        
        # Compute derivatives for mean flow
        dtdt = torch.ones_like(t)
        drdt = torch.zeros_like(r)
        
        with torch.amp.autocast("cuda", enabled=False):
            # Compute predicted velocity and its derivative
            u_pred, dudt = torch.func.jvp(u_func, (z, t, r), (v, dtdt, drdt))
            
            # Target velocity field
            u_tgt = (v - (t - r) * dudt).detach()
            
            # Mean flow reconstruction loss
            flow_loss = (u_pred - u_tgt)**2
            flow_loss = flow_loss.mean(dim=(1, 2))  # Mean over channel and time dimensions
            
            # Adaptive weighting for stability
            adp_wt = (flow_loss.detach() + self.args.norm_eps) ** self.args.norm_p
            flow_loss = flow_loss / adp_wt
            flow_loss = flow_loss.mean()  # Mean over batch
        
        # Direct denoising loss: predict clean from noisy at t=0.5
        # This helps the model learn the direct noisy->clean mapping
        with torch.no_grad():
            t_mid = torch.full((batch_size,), 0.5, device=device)
            t_mid_expanded = t_mid.view(-1, 1, 1)
            e_fixed = torch.randn_like(x_clean)
            z_mid = (1 - t_mid_expanded) * x_clean + t_mid_expanded * e_fixed
        
        u_mid = self.net(
            z_mid,
            (t_mid, t_mid),  # t=r for single-step inference
            aug_cond,
            class_labels=class_labels,
            noisy_cond=x_noisy
        )
        x_pred = z_mid - t_mid_expanded * u_mid
        denoising_loss = F.mse_loss(x_pred, x_clean)
        
        # Total loss
        total_loss = flow_loss + 0.5 * denoising_loss
        
        return {
            'total_loss': total_loss,
            'flow_loss': flow_loss,
            'denoising_loss': denoising_loss
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
        
        For mean flow with single-step inference (t=1, r=0):
        x_clean = z_1 - u(z_1, t=1, h=1)
        
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
            # Single-step denoising using mean flow
            # Start from noise interpolated with noisy signal
            e = torch.randn_like(x_noisy)
            t = torch.ones(batch_size, device=device)
            
            # For denoising, we start from z_1 (mixture of target and noise)
            # But we condition on the noisy signal
            z_1 = e  # Start from pure noise
            
            # Predict velocity
            u = net(z_1, (t, t), aug_cond=None, class_labels=class_labels, noisy_cond=x_noisy)
            
            # Clean signal estimate
            x_clean = z_1 - u
        else:
            # Multi-step denoising
            z = torch.randn_like(x_noisy)
            dt = 1.0 / num_steps
            
            for step in range(num_steps):
                t_val = 1.0 - step * dt
                t = torch.full((batch_size,), t_val, device=device)
                
                # Predict velocity
                u = net(z, (t, t), aug_cond=None, class_labels=class_labels, noisy_cond=x_noisy)
                
                # Euler step
                z = z - dt * u
            
            x_clean = z
        
        return x_clean
    
    def sample(
        self, 
        samples_shape: Tuple[int, ...],
        class_labels: Optional[torch.Tensor] = None,
        noisy_cond: Optional[torch.Tensor] = None,
        net: Optional[nn.Module] = None,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Sample from the model (generate clean signals)
        
        Args:
            samples_shape: Shape of samples to generate
            class_labels: Class labels for conditional generation
            noisy_cond: Noisy signal for conditioning denoising
            net: Network to use (default: self.net_ema)
            device: Device to generate on
        
        Returns:
            Generated/denoised samples
        """
        net = net if net is not None else self.net_ema
        batch_size = samples_shape[0]
        
        # Sample initial noise
        e = torch.randn(samples_shape, dtype=torch.float32, device=device)
        z_1 = e
        
        # Set time steps
        t = torch.ones(batch_size, device=device)
        r = torch.zeros(batch_size, device=device)
        
        # Generate with class conditioning
        if class_labels is not None:
            u = net(z_1, (t, t - r), aug_cond=None, class_labels=class_labels, noisy_cond=noisy_cond)
        else:
            # Unconditional generation (sample random classes)
            random_labels = torch.randint(0, self.num_classes, (batch_size,), device=device)
            u = net(z_1, (t, t - r), aug_cond=None, class_labels=random_labels, noisy_cond=noisy_cond)
        
        # Final sample
        z_0 = z_1 - u
        
        return z_0

