"""
UNet Architecture for Signal Denoising
Adapted from ModulationUNet with noisy signal conditioning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import silu
import math
from typing import Optional, Tuple

from .unet import (
    weight_init, Linear, GroupNorm, 
    PositionalEmbedding, FourierEmbedding
)
from .unet_modulation import Conv1d, ModulationUNetBlock, ModulationAttention


class DenoisingUNet(nn.Module):
    """
    UNet architecture for signal denoising
    Takes noisy signal as conditioning input
    Learns to predict clean signal (x-prediction)
    """
    
    def __init__(
        self,
        signal_length: int = 128,  # Length of I/Q signal
        in_channels: int = 2,  # I and Q channels
        out_channels: int = 2,  # Output I and Q channels
        num_classes: int = 9,  # Number of modulation classes
        model_channels: int = 64,  # Base channel dimension
        channel_mult: Tuple[int, ...] = (1, 2, 2, 2),  # Channel multipliers
        num_blocks: int = 2,  # Blocks per resolution
        dropout: float = 0.1,  # Dropout probability
        class_dropout: float = 0.1,  # Class label dropout for CFG
        use_attention: bool = True,  # Whether to use attention
        attention_levels: Tuple[int, ...] = (2, 3),  # Which levels to apply attention
        embedding_type: str = 'positional',  # Time embedding type
        class_embed_dim: Optional[int] = None  # Class embedding dimension
    ):
        """
        Initialize DenoisingUNet
        
        Args:
            signal_length: Length of input signal
            in_channels: Number of input channels (2 for I/Q)
            out_channels: Number of output channels
            num_classes: Number of modulation classes
            model_channels: Base channel dimension
            channel_mult: Channel multipliers per level
            num_blocks: Number of blocks per level
            dropout: Dropout probability
            class_dropout: Class dropout for classifier-free guidance
            use_attention: Whether to use self-attention
            attention_levels: Which levels to use attention
            embedding_type: Type of time embedding
            class_embed_dim: Dimension of class embedding
        """
        super().__init__()
        
        self.signal_length = signal_length
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.class_dropout = class_dropout
        
        # Time embedding dimension
        time_embed_dim = model_channels * 4
        
        # Class embedding dimension
        if class_embed_dim is None:
            class_embed_dim = model_channels * 2
        self.class_embed_dim = class_embed_dim
        
        # Time embedding
        if embedding_type == 'positional':
            self.time_embed = nn.Sequential(
                PositionalEmbedding(num_channels=model_channels),
                Linear(model_channels, time_embed_dim),
                nn.SiLU(),
                Linear(time_embed_dim, time_embed_dim)
            )
        else:
            self.time_embed = nn.Sequential(
                FourierEmbedding(num_channels=model_channels),
                Linear(model_channels, time_embed_dim),
                nn.SiLU(),
                Linear(time_embed_dim, time_embed_dim)
            )
        
        # Class embedding
        self.class_embed = nn.Embedding(num_classes, class_embed_dim)
        self.class_proj = Linear(class_embed_dim, time_embed_dim)
        # Dedicated null-class embedding for classifier-free guidance
        self.null_class_time = nn.Parameter(torch.zeros(time_embed_dim))
        
        # For denoising, we don't need separate noisy conditioning
        # The noisy signal is the starting point (t=1) of the flow
        init_in_channels = in_channels
        
        # Initial convolution
        self.init_conv = Conv1d(
            in_channels=init_in_channels,
            out_channels=model_channels,
            kernel_size=3,
            padding=1
        )
        
        # Encoder blocks
        self.encoder = nn.ModuleList()
        channels = [model_channels]
        now_channels = model_channels
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            for _ in range(num_blocks):
                self.encoder.append(
                    ModulationUNetBlock(
                        in_channels=now_channels,
                        out_channels=out_ch,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        use_attention=(use_attention and level in attention_levels)
                    )
                )
                now_channels = out_ch
                channels.append(now_channels)
            
            # Downsample (except last level)
            if level < len(channel_mult) - 1:
                self.encoder.append(
                    Conv1d(
                        in_channels=now_channels,
                        out_channels=now_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1
                    )
                )
                channels.append(now_channels)
        
        # Middle blocks
        self.middle = nn.ModuleList([
            ModulationUNetBlock(
                in_channels=now_channels,
                out_channels=now_channels,
                emb_channels=time_embed_dim,
                dropout=dropout,
                use_attention=True
            ),
            ModulationUNetBlock(
                in_channels=now_channels,
                out_channels=now_channels,
                emb_channels=time_embed_dim,
                dropout=dropout,
                use_attention=False
            )
        ])
        
        # Decoder blocks
        self.decoder = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            
            # Upsample (except first level in reversed order)
            if level < len(channel_mult) - 1:
                self.decoder.append(
                    nn.ConvTranspose1d(
                        in_channels=now_channels,
                        out_channels=now_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1
                    )
                )
            
            for _ in range(num_blocks + 1):
                skip_channels = channels.pop()
                
                self.decoder.append(
                    ModulationUNetBlock(
                        in_channels=now_channels + skip_channels,
                        out_channels=out_ch,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        use_attention=(use_attention and level in attention_levels)
                    )
                )
                now_channels = out_ch
        
        # Final output layers
        self.final_norm = GroupNorm(num_channels=now_channels)
        self.final_conv = Conv1d(
            in_channels=now_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1,
            init_weight=0.0
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        aug_cond: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        noisy_cond: Optional[torch.Tensor] = None,
        snr_values: Optional[torch.Tensor] = None  # Not used, for API compatibility
    ) -> torch.Tensor:
        """
        Forward pass through the denoising UNet
        
        Args:
            x: Input signal [batch, 2, 128] (diffusion state z_t)
            t: Time conditioning tensor [batch]
            aug_cond: Augmentation conditioning (optional)
            class_labels: Class labels for modulation conditioning
            noisy_cond: Noisy signal for conditioning denoising
            snr_values: SNR values (not used, kept for API compatibility)
        
        Returns:
            Output clean signal prediction [batch, 2, 128]
        """
        # Ensure correct shape [batch, 2, 128]
        if x.dim() == 3 and x.shape[1] == 2 and x.shape[2] == 128:
            pass
        else:
            batch_size = x.shape[0]
            x = x.view(batch_size, 2, 128)
        
        # Compute time embedding
        time_emb = self.time_embed(t)
        
        # Add class embedding if provided
        if class_labels is not None:
            class_emb = self.class_embed(class_labels)  
            class_emb = self.class_proj(class_emb)
            
            # Apply classifier-free guidance
            if self.training and self.class_dropout > 0:
                drop_mask = (torch.rand(class_labels.shape[0], device=x.device) < self.class_dropout)
                if drop_mask.any():
                    class_emb = class_emb.clone()
                    null_row = self.null_class_time.to(dtype=class_emb.dtype, device=class_emb.device).unsqueeze(0)
                    class_emb[drop_mask] = null_row.expand(class_emb[drop_mask].shape[0], -1)
            time_emb = time_emb + class_emb
        
        # For denoising, x itself is the noisy signal at t=1 or interpolated state
        # No need for separate noisy conditioning
        x_input = x
        
        # Apply SiLU activation to combined embedding
        emb = silu(time_emb)
        
        # Initial convolution
        h_state = self.init_conv(x_input)
        
        # Encoder
        skips = [h_state]
        for module in self.encoder:
            if isinstance(module, ModulationUNetBlock):
                h_state = module(h_state, emb)
                skips.append(h_state)
            else:
                # Downsampling convolution
                h_state = module(h_state)
                skips.append(h_state)
        
        # Middle blocks
        for module in self.middle:
            h_state = module(h_state, emb)
        
        # Decoder
        for module in self.decoder:
            if isinstance(module, nn.ConvTranspose1d):
                h_state = module(h_state)
            else:
                skip = skips.pop()
                h_state = torch.cat([h_state, skip], dim=1)
                h_state = module(h_state, emb)
        
        # Final output
        h_state = self.final_norm(h_state)
        h_state = silu(h_state)
        h_state = self.final_conv(h_state)
        
        # Ensure output shape matches expected
        if h_state.shape[1:] != (2, 128):
            h_state = h_state.view(x.shape[0], 2, 128)
        
        return h_state

