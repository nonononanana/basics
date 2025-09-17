"""
Mean Flow Model for Open-Set Modulation Classification
Implements conditional generation with modulation type as condition
Energy-based OOD detection for unknown modulations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List

from models.time_sampler import sample_two_timesteps
from models.ema import init_ema, update_ema_net
import logging

logger = logging.getLogger(__name__)


class MeanFlowModulation(nn.Module):
    """
    Mean Flow model adapted for modulation classification
    Uses modulation type as condition for generation
    """
    
    def __init__(
        self, 
        arch,  # UNet architecture
        args,  # Training arguments
        net_configs,  # Network configuration
        num_classes: int = 9,  # Number of known modulation classes
        class_embed_dim: int = 128,  # Dimension of class embedding
        use_arcface: bool = True,  # Whether to use ArcFace loss
        arcface_margin: float = 0.3,  # ArcFace margin parameter
        arcface_scale: float = 30.0,  # ArcFace scale parameter
        energy_temperature: float = 1.0  # Temperature for energy scoring
    ):
        """
        Initialize Mean Flow model for modulation classification
        
        Args:
            arch: Neural network architecture (e.g., UNet)
            args: Training arguments
            net_configs: Network configuration dict
            num_classes: Number of known modulation classes
            class_embed_dim: Dimension of class embedding
            use_arcface: Whether to use ArcFace for better class separation
            arcface_margin: Margin parameter for ArcFace
            arcface_scale: Scale parameter for ArcFace
            energy_temperature: Temperature for energy-based OOD detection
        """
        super(MeanFlowModulation, self).__init__()
        
        self.num_classes = num_classes
        self.class_embed_dim = class_embed_dim
        self.use_arcface = use_arcface
        self.arcface_margin = arcface_margin
        self.arcface_scale = arcface_scale
        self.energy_temperature = energy_temperature
        self.args = args
        
        # Main network with class conditioning
        self.net = arch(**net_configs)
        
        # Class embedding layer for conditioning
        # Maps class labels to embedding vectors
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
        
        # ArcFace components if enabled
        if self.use_arcface:
            # ArcFace weight matrix for angular margin
            self.arcface_weight = nn.Parameter(
                torch.FloatTensor(num_classes, class_embed_dim)
            )
            nn.init.xavier_uniform_(self.arcface_weight)
            
            # Pre-compute cos(margin) and sin(margin)
            self.cos_m = np.cos(arcface_margin)
            self.sin_m = np.sin(arcface_margin)
            self.threshold = np.cos(np.pi - arcface_margin)
            self.mm = np.sin(np.pi - arcface_margin) * arcface_margin
    
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
    
    def arcface_loss(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ArcFace loss for better class separability
        
        Args:
            features: Feature embeddings [batch_size, embed_dim]
            labels: Class labels [batch_size]
        
        Returns:
            ArcFace loss value
        """
        # Normalize features and weights
        features = F.normalize(features, p=2, dim=1)
        weights = F.normalize(self.arcface_weight, p=2, dim=1)
        
        # Compute cosine similarity
        cos_theta = F.linear(features, weights)  # [batch_size, num_classes]
        cos_theta = cos_theta.clamp(-1, 1)  # Numerical stability
        
        # Get angle
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        
        # Apply angular margin
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        
        # Apply threshold to prevent gradient explosion
        cos_theta_m = torch.where(
            cos_theta > self.threshold,
            cos_theta_m,
            cos_theta - self.mm
        )
        
        # Create one-hot labels
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Apply margin only to correct class
        output = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        
        # Scale and compute cross-entropy loss
        output = output * self.arcface_scale
        
        return F.cross_entropy(output, labels)
    
    def forward_with_loss(
        self, 
        x_pos: torch.Tensor,
        x_neg: Optional[torch.Tensor], 
        class_labels: torch.Tensor,
        aug_cond: Optional[torch.Tensor] = None,
        lambda_rec: float = 1.0,
        lambda_arc: float = 5.0,
        lambda_pos: float = 0.1,
        lambda_neg: float = 0.5,
        margin_pos: float = 0.1,
        margin_neg: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with loss computation including energy-based losses
        
        Args:
            x_pos: Positive (original) signal data [batch_size, 2, 128] (I/Q channels)
            x_neg: Negative (corrupted) signal data [batch_size, 2, 128] or None
            class_labels: Modulation class labels [batch_size]
            aug_cond: Optional augmentation conditioning
            lambda_rec: Weight for reconstruction loss
            lambda_arc: Weight for ArcFace loss
            lambda_pos: Weight for positive energy loss
            lambda_neg: Weight for negative energy loss
            margin_pos: Margin for positive energy (should be small)
            margin_neg: Margin for negative energy (should be large)
        
        Returns:
            Dictionary containing different loss components
        """
        device = x_pos.device
        batch_size = x_pos.shape[0]
        
        # Sample noise for diffusion
        e = torch.randn_like(x_pos).to(device)
        
        # Sample two timesteps for mean flow
        t, r = sample_two_timesteps(self.args, num_samples=batch_size, device=device)
        # Reshape for 1D signal broadcasting [batch, 1, 1]
        t = t.view(-1, 1, 1)
        r = r.view(-1, 1, 1)
        
        # Interpolate between data and noise for positive samples
        z = (1 - t) * x_pos + t * e
        v = e - x_pos  # Velocity field
        
        # Get class embeddings for conditioning
        class_embeds = self.get_class_embeddings(class_labels)
        
        # Define network function with class conditioning
        def u_func(z, t, r):
            h = t - r
            # Pass class labels to network for conditioning
            return self.net(
                z, 
                (t.view(-1), h.view(-1)), 
                aug_cond,
                class_labels=class_labels  # Pass class labels for conditioning
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
            reconstruction_loss = (u_pred - u_tgt)**2
            reconstruction_loss = reconstruction_loss.mean(dim=(1, 2))  # Mean over channel and time dimensions
            
            # Adaptive weighting for stability
            adp_wt = (reconstruction_loss.detach() + self.args.norm_eps) ** self.args.norm_p
            reconstruction_loss = reconstruction_loss / adp_wt
            reconstruction_loss = reconstruction_loss.mean()  # Mean over batch
        
        # Compute ArcFace loss if enabled and we have valid labels
        arcface_loss = torch.tensor(0.0, device=device)
        if self.use_arcface and (class_labels >= 0).any():
            # Extract features from intermediate representation
            # Use class embeddings as features for ArcFace
            valid_mask = class_labels >= 0
            if valid_mask.any():
                valid_features = class_embeds[valid_mask]
                valid_labels = class_labels[valid_mask]
                arcface_loss = self.arcface_loss(valid_features, valid_labels)
        
        # Compute energy losses for positive samples
        pos_energy_loss = torch.tensor(0.0, device=device)
        if lambda_pos > 0:
            with torch.no_grad():
                # Compute energy score for positive samples
                pos_energies = self.compute_energy_score(x_pos, return_per_class=False)
            
            # L_pos = max(0, E(x) - m_pos)
            # We want positive samples to have low energy
            pos_energy_loss = F.relu(pos_energies - margin_pos).mean()
        
        # Compute energy losses for negative samples
        neg_energy_loss = torch.tensor(0.0, device=device)
        if lambda_neg > 0 and x_neg is not None:
            with torch.no_grad():
                # Compute energy score for negative samples
                neg_energies = self.compute_energy_score(x_neg, return_per_class=False)
            
            # L_neg = max(0, m_neg - E(x))
            # We want negative samples to have high energy
            neg_energy_loss = F.relu(margin_neg - neg_energies).mean()
        
        # Total loss with weighted components
        total_loss = (
            lambda_rec * reconstruction_loss +
            lambda_arc * arcface_loss +
            lambda_pos * pos_energy_loss +
            lambda_neg * neg_energy_loss
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'arcface_loss': arcface_loss,
            'pos_energy_loss': pos_energy_loss,
            'neg_energy_loss': neg_energy_loss
        }
    
    def sample(
        self, 
        samples_shape: Tuple[int, ...],
        class_labels: Optional[torch.Tensor] = None,
        net: Optional[nn.Module] = None,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Sample from the model (generate modulation signals)
        
        Args:
            samples_shape: Shape of samples to generate
            class_labels: Class labels for conditional generation
            net: Network to use (default: self.net_ema)
            device: Device to generate on
        
        Returns:
            Generated samples
        """
        net = net if net is not None else self.net_ema
        batch_size = samples_shape[0]
        
        # Sample initial noise
        e = torch.randn(samples_shape, dtype=torch.float32, device=device)
        z_1 = e
        
        # Set time steps
        t = torch.ones(batch_size, device=device)
        r = torch.zeros(batch_size, device=device)
        
        # Generate with class conditioning if provided
        if class_labels is not None:
            u = net(z_1, (t, t - r), aug_cond=None, class_labels=class_labels)
        else:
            # Unconditional generation (sample random classes)
            random_labels = torch.randint(0, self.num_classes, (batch_size,), device=device)
            u = net(z_1, (t, t - r), aug_cond=None, class_labels=random_labels)
        
        # Final sample
        z_0 = z_1 - u
        
        return z_0
    
    def compute_energy_score(
        self, 
        x: torch.Tensor,
        return_per_class: bool = False
    ) -> torch.Tensor:
        """
        Compute energy score for OOD detection
        Lower energy indicates in-distribution (known class)
        Higher energy indicates out-of-distribution (unknown class)
        
        Args:
            x: Input signal [batch_size, 2, 128]
            return_per_class: If True, return energy for each class
        
        Returns:
            Energy scores [batch_size] or [batch_size, num_classes]
        """
        batch_size = x.shape[0]
        device = x.device
        
        if return_per_class:
            # Compute energy for each class
            energies = []
            
            for class_idx in range(self.num_classes):
                # Create labels for this class
                class_labels = torch.full((batch_size,), class_idx, dtype=torch.long, device=device)
                
                # Compute reconstruction under this class assumption
                with torch.no_grad():
                    # Sample noise
                    e = torch.randn_like(x)
                    
                    # Use intermediate time for energy computation
                    t = torch.full((batch_size,), 0.5, device=device)
                    t_expanded = t.view(-1, 1, 1)
                    
                    # Interpolate
                    z = (1 - t_expanded) * x + t_expanded * e
                    
                    # Predict velocity field
                    u = self.net_ema(
                        z,
                        (t, t),
                        aug_cond=None,
                        class_labels=class_labels
                    )
                    
                    # Compute reconstruction
                    x_recon = z - t_expanded * u
                    
                    # Compute reconstruction error as energy
                    error = (x - x_recon)**2
                    error = error.mean(dim=(1, 2))  # Mean over channel and time dimensions
                    
                    energies.append(error)
            
            # Stack energies for all classes
            energies = torch.stack(energies, dim=1)  # [batch_size, num_classes]
            
            # Apply temperature scaling and compute log-sum-exp
            # This gives us the energy score
            energy_score = -self.energy_temperature * torch.logsumexp(
                -energies / self.energy_temperature, dim=1
            )
            
            if return_per_class:
                return energies
            else:
                return energy_score
        else:
            # Compute minimum energy across all classes
            energies = self.compute_energy_score(x, return_per_class=True)
            
            # Return negative log-sum-exp as energy score
            energy_score = -self.energy_temperature * torch.logsumexp(
                -energies / self.energy_temperature, dim=1
            )
            
            return energy_score
    
    def classify_with_rejection(
        self,
        x: torch.Tensor,
        energy_threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Classify input with rejection option for unknown classes
        
        Args:
            x: Input signal [batch_size, 2, 128]
            energy_threshold: Threshold for rejection (higher = more rejection)
        
        Returns:
            predictions: Predicted class labels (-1 for rejected/unknown)
            energies: Energy scores for each sample
            class_energies: Energy for each class [batch_size, num_classes]
        """
        # Compute energy for each class
        class_energies = self.compute_energy_score(x, return_per_class=True)
        
        # Find minimum energy class (best match)
        min_energies, predictions = class_energies.min(dim=1)
        
        # Compute overall energy score
        energy_scores = -self.energy_temperature * torch.logsumexp(
            -class_energies / self.energy_temperature, dim=1
        )
        
        # Reject samples with high energy (unknown classes)
        # Normalize energy threshold based on data statistics
        reject_mask = energy_scores > energy_threshold
        predictions[reject_mask] = -1  # Mark as unknown
        
        return predictions, energy_scores, class_energies
