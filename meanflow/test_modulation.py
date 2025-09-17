"""
Quick test script to verify the modulation classification implementation
Tests data loading, model creation, and basic forward pass
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from meanflow.data.rml_dataset import RML2016Dataset, get_rml_dataloaders, EXPERIMENT_SETTINGS
from meanflow.models.meanflow_modulation import MeanFlowModulation
from meanflow.models.unet_modulation import ModulationUNet


def test_data_loading():
    """Test dataset loading and structure"""
    print("=" * 50)
    print("Testing Data Loading...")
    print("=" * 50)
    
    # Check if data file exists
    data_path = Path("data/RML2016.10a_dict.pkl")
    if not data_path.exists():
        print(f"Data file not found at {data_path}")
        print("Please ensure RML2016.10a_dict.pkl is in the data/ directory")
        return False
    
    try:
        # Test dataset creation for experiment setting 1
        train_dataset = RML2016Dataset(
            data_path=str(data_path),
            experiment_setting=1,
            train=True,
            snr_range=(-20, 20),
            normalize=True,
            augment=False,
            return_snr=True,
            seed=42
        )
        
        test_dataset = RML2016Dataset(
            data_path=str(data_path),
            experiment_setting=1,
            train=False,
            snr_range=(-20, 20),
            normalize=True,
            augment=False,
            return_snr=True,
            seed=42
        )
        
        print(f"✓ Train dataset size: {len(train_dataset)}")
        print(f"✓ Test dataset size: {len(test_dataset)}")
        
        # Get a sample
        sample, label, info = train_dataset[0]
        print(f"✓ Sample shape: {sample.shape}")
        print(f"✓ Label: {label}")
        print(f"✓ Is unknown: {info['is_unknown']}")
        print(f"✓ SNR: {info['snr']}")
        
        # Test dataloader creation
        train_loader, test_loader = get_rml_dataloaders(
            data_path=str(data_path),
            experiment_setting=1,
            batch_size=32,
            num_workers=0,  # Use 0 for testing
            snr_range=(-20, 20),
            train_split=0.8,
            normalize=True,
            augment_train=False,
            seed=42
        )
        
        print(f"✓ Train loader batches: {len(train_loader)}")
        print(f"✓ Test loader batches: {len(test_loader)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in data loading: {e}")
        return False


def test_model_creation():
    """Test model instantiation"""
    print("\n" + "=" * 50)
    print("Testing Model Creation...")
    print("=" * 50)
    
    try:
        # Create dummy args object
        class Args:
            ema_decay = 0.999
            ema_decays = [0.9999]
            norm_eps = 1e-5
            norm_p = -0.5
            num_timesteps = 1000
            tr_sampler = 'v1'  # Time sampler version
            tr_epsilon = 1e-5  # Epsilon for time sampling
            P_mean_t = 0.0  # Mean for t sampling
            P_std_t = 1.0  # Std for t sampling
            P_mean_r = 0.0  # Mean for r sampling
            P_std_r = 1.0  # Std for r sampling
            ratio = 0.5  # Ratio for time sampling
        
        args = Args()
        
        # Get number of classes for experiment 1
        num_classes = len(EXPERIMENT_SETTINGS[1]['known'])
        print(f"Number of known classes: {num_classes}")
        
        # Create UNet configuration
        net_configs = {
            'signal_length': 128,
            'in_channels': 2,
            'out_channels': 2,
            'num_classes': num_classes,
            'model_channels': 32,  # Small for testing
            'channel_mult': (1, 2, 2),
            'num_blocks': 1,  # Minimal for testing
            'dropout': 0.1,
            'class_dropout': 0.1,
            'use_attention': True,
            'attention_levels': (2,),
            'embedding_type': 'positional',
            'class_embed_dim': 64
        }
        
        # Create model
        model = MeanFlowModulation(
            arch=ModulationUNet,
            args=args,
            net_configs=net_configs,
            num_classes=num_classes,
            use_arcface=True,
            arcface_margin=0.5,
            arcface_scale=30.0,
            energy_temperature=1.0
        )
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Model created with {num_params:,} parameters")
        
        # Test model device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        print(f"✓ Model moved to {device}")
        
        return True, model, args, device
        
    except Exception as e:
        print(f"✗ Error in model creation: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None


def test_forward_pass(model, args, device):
    """Test model forward pass"""
    print("\n" + "=" * 50)
    print("Testing Forward Pass...")
    print("=" * 50)
    
    try:
        # Create dummy batch
        batch_size = 4
        signals = torch.randn(batch_size, 2, 128).to(device)  # I/Q signals
        labels = torch.randint(0, model.num_classes, (batch_size,)).to(device)
        
        print(f"Input shape: {signals.shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Test forward with loss
        model.train()
        loss_dict = model.forward_with_loss(
            x=signals,
            class_labels=labels,
            aug_cond=None
        )
        
        print(f"✓ Total loss: {loss_dict['total_loss'].item():.4f}")
        print(f"✓ Reconstruction loss: {loss_dict['reconstruction_loss'].item():.4f}")
        print(f"✓ ArcFace loss: {loss_dict['arcface_loss'].item():.4f}")
        
        # Test energy scoring
        model.eval()
        with torch.no_grad():
            energy_scores = model.compute_energy_score(signals)
            print(f"✓ Energy scores shape: {energy_scores.shape}")
            print(f"✓ Energy scores: {energy_scores.cpu().numpy()}")
            
            # Test classification with rejection
            predictions, energies, class_energies = model.classify_with_rejection(
                signals, energy_threshold=0.5
            )
            print(f"✓ Predictions: {predictions.cpu().numpy()}")
            
            # Test generation
            generated = model.sample(
                samples_shape=(2, 2, 128),
                class_labels=torch.tensor([0, 1], device=device),
                device=device
            )
            print(f"✓ Generated shape: {generated.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("MEAN FLOW MODULATION CLASSIFICATION - TEST SUITE")
    print("=" * 70)
    
    # Test data loading
    data_ok = test_data_loading()
    if not data_ok:
        print("\n⚠ Data loading failed. Please check data files.")
        return
    
    # Test model creation
    model_ok, model, args, device = test_model_creation()
    if not model_ok:
        print("\n⚠ Model creation failed.")
        return
    
    # Test forward pass
    forward_ok = test_forward_pass(model, args, device)
    if not forward_ok:
        print("\n⚠ Forward pass failed.")
        return
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
    print("\nThe implementation is ready for training.")
    print("Run: python meanflow/train_modulation.py --help")
    print("Or use the provided script: bash meanflow/scripts/run_modulation_experiments.sh")


if __name__ == "__main__":
    main()
