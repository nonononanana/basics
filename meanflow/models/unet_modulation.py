"""
UNet Architecture for Modulation Signal Processing
Adapted for I/Q signal data with class conditioning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import silu
import math
from typing import Optional, Tuple

from .groupnorm import group_norm
from .unet import (
    weight_init, Linear, Conv2d, GroupNorm, 
    PositionalEmbedding, FourierEmbedding, QKVAttention
)


class Conv1d(nn.Module):
    """1D Convolution layer for signal processing"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        init_mode: str = 'kaiming_normal',
        init_weight: float = 1.0,
        init_bias: float = 0.0
    ):
        """
        Initialize 1D convolution layer
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel
            stride: Stride of convolution
            padding: Padding size
            bias: Whether to use bias
            init_mode: Weight initialization mode
            init_weight: Weight initialization scale
            init_bias: Bias initialization scale
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initialize weights
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel_size,
            fan_out=out_channels * kernel_size
        )
        
        # Create weight parameter
        self.weight = nn.Parameter(
            weight_init([out_channels, in_channels, kernel_size], **init_kwargs) * init_weight
        )
        
        # Create bias parameter if needed
        self.bias = nn.Parameter(
            weight_init([out_channels], **init_kwargs) * init_bias
        ) if bias else None
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, channels, length]
        
        Returns:
            Output tensor
        """
        # Apply 1D convolution
        x = F.conv1d(
            x,
            self.weight.to(x.dtype),
            self.bias.to(x.dtype) if self.bias is not None else None,
            stride=self.stride,
            padding=self.padding
        )
        return x


class ModulationUNetBlock(nn.Module):
    """
    UNet block for modulation signal processing
    Processes 1D I/Q signals with residual connections
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        dropout: float = 0.0,
        skip_scale: float = 1.0,
        eps: float = 1e-5,
        use_attention: bool = False,
        num_heads: int = 4
    ):
        """
        Initialize UNet block for modulation signals
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            emb_channels: Number of embedding channels
            dropout: Dropout probability
            skip_scale: Scale factor for skip connections
            eps: Epsilon for normalization
            use_attention: Whether to use self-attention
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.use_attention = use_attention
        
        # First convolution block
        self.norm1 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv1 = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        
        # Time/class embedding projection
        self.emb_proj = Linear(
            in_features=emb_channels,
            out_features=out_channels * 2  # For scale and shift
        )
        
        # Second convolution block
        self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv2 = Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            init_weight=0.0  # Initialize to zero for residual
        )
        
        # Skip connection if dimensions change
        self.skip = None
        if in_channels != out_channels:
            self.skip = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0
            )
        
        # Self-attention if enabled
        if use_attention:
            self.attention = ModulationAttention(
                channels=out_channels,
                num_heads=num_heads,
                eps=eps
            )
    
    def forward(
        self, 
        x: torch.Tensor, 
        emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, channels, length]
            emb: Embedding tensor [batch, emb_channels]
        
        Returns:
            Output tensor [batch, out_channels, length]
        """
        # Store input for skip connection
        h = x
        
        # First convolution with normalization and activation
        x = self.norm1(x)
        x = silu(x)
        x = self.conv1(x)
        
        # Add time/class embedding
        emb_out = self.emb_proj(emb)
        emb_out = emb_out.unsqueeze(-1)  # Add length dimension
        scale, shift = emb_out.chunk(2, dim=1)
        
        # Apply scale and shift from embedding
        x = self.norm2(x)
        x = x * (1 + scale) + shift
        x = silu(x)
        
        # Second convolution with dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x)
        
        # Skip connection
        if self.skip is not None:
            h = self.skip(h)
        
        x = (x + h) * self.skip_scale
        
        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x)
        
        return x


class ModulationAttention(nn.Module):
    """
    Self-attention module for modulation signals
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        eps: float = 1e-5
    ):
        """
        Initialize attention module
        
        Args:
            channels: Number of channels
            num_heads: Number of attention heads
            eps: Epsilon for normalization
        """
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        # Normalization
        self.norm = GroupNorm(num_channels=channels, eps=eps)
        
        # QKV projection
        self.qkv = Conv1d(
            in_channels=channels,
            out_channels=channels * 3,
            kernel_size=1,
            padding=0
        )
        
        # Output projection
        self.proj = Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            padding=0,
            init_weight=0.0
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention
        
        Args:
            x: Input tensor [batch, channels, length]
        
        Returns:
            Output tensor with attention applied
        """
        h = x
        x = self.norm(x)
        
        # Compute QKV
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        batch_size, _, length = q.shape
        q = q.reshape(batch_size, self.num_heads, self.head_dim, length)
        k = k.reshape(batch_size, self.num_heads, self.head_dim, length)
        v = v.reshape(batch_size, self.num_heads, self.head_dim, length)
        
        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.einsum('bhcl,bhcm->bhlm', q, k) * scale
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.einsum('bhlm,bhcm->bhcl', attn, v)
        out = out.reshape(batch_size, self.channels, length)
        
        # Output projection and residual
        out = self.proj(out)
        out = out + h
        
        return out


class ModulationUNet(nn.Module):
    """
    UNet architecture for modulation signal processing
    Handles I/Q signals with class conditioning
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
        class_embed_dim: int = 128  # Class embedding dimension
    ):
        """
        Initialize ModulationUNet
        
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
        # Dedicated null-class time embedding to avoid colliding with real class 0
        self.null_class_time = nn.Parameter(torch.zeros(time_embed_dim))
        
        # Initial convolution
        self.init_conv = Conv1d(
            in_channels=in_channels,
            out_channels=model_channels,
            kernel_size=3,
            padding=1
        )
        
        # Encoder blocks
        self.encoder = nn.ModuleList()
        channels = [model_channels]
        now_channels = model_channels
        
        for level, mult in enumerate(channel_mult):
            out_channels = model_channels * mult
            
            for _ in range(num_blocks):
                # Add residual block
                self.encoder.append(
                    ModulationUNetBlock(
                        in_channels=now_channels,
                        out_channels=out_channels,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        use_attention=(use_attention and level in attention_levels)
                    )
                )
                now_channels = out_channels
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
            out_channels = model_channels * mult
            
            # Upsample (except first level)
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
                # Pop skip connection channels
                skip_channels = channels.pop()
                
                # Add residual block with skip connection
                self.decoder.append(
                    ModulationUNetBlock(
                        in_channels=now_channels + skip_channels,
                        out_channels=out_channels,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        use_attention=(use_attention and level in attention_levels)
                    )
                )
                now_channels = out_channels
        
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
        time_cond: Tuple[torch.Tensor, torch.Tensor],
        aug_cond: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the UNet
        
        Args:
            x: Input signal [batch, 2, 128] (I/Q channels)
            time_cond: Time conditioning (t, h) tuple
            aug_cond: Augmentation conditioning (optional)
            class_labels: Class labels for conditioning
        
        Returns:
            Output signal [batch, 2, 128]
        """
        # Reshape if needed (ensure we have [batch, channels, length])
        if x.dim() == 3 and x.shape[1] == 2 and x.shape[2] == 128:
            # Already in correct format
            pass
        else:
            # Might need to reshape
            batch_size = x.shape[0]
            x = x.view(batch_size, 2, 128)
        
        # Extract time conditions
        t, h = time_cond
        
        # Compute time embedding
        # Combine t and h for time embedding
        time_emb = self.time_embed(t) + self.time_embed(h)
        
        # Add class embedding if provided
        if class_labels is not None:
            # Compute class-conditioned contribution
            class_emb = self.class_embed(class_labels)
            class_emb = self.class_proj(class_emb)
            # Apply classifier-free guidance via a dedicated null embedding, not class 0
            if self.training and self.class_dropout > 0:
                drop_mask = torch.rand(class_labels.shape[0], device=x.device) < self.class_dropout
                if drop_mask.any():
                    null_add = self.null_class_time.unsqueeze(0).expand_as(class_emb)
                    class_emb = class_emb.clone()
                    class_emb[drop_mask] = null_add[drop_mask]
            # Add to time embedding
            time_emb = time_emb + class_emb
        
        # Apply SiLU activation to combined embedding
        emb = silu(time_emb)
        
        # Initial convolution
        h = self.init_conv(x)
        
        # Encoder
        skips = [h]
        for module in self.encoder:
            if isinstance(module, ModulationUNetBlock):
                h = module(h, emb)
                skips.append(h)
            else:
                # Downsampling convolution
                h = module(h)
                skips.append(h)
        
        # Middle blocks
        for module in self.middle:
            h = module(h, emb)
        
        # Decoder
        for module in self.decoder:
            if isinstance(module, nn.ConvTranspose1d):
                # Upsampling
                h = module(h)
            else:
                # UNet block with skip connection
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = module(h, emb)
        
        # Final output
        h = self.final_norm(h)
        h = silu(h)
        h = self.final_conv(h)
        
        # Ensure output shape matches input
        if h.shape != x.shape:
            h = h.view(x.shape)
        
        return h
