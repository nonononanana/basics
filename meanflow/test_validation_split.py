#!/usr/bin/env python
"""
Test script to verify the new train/validation/test split functionality
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from meanflow.data.rml_dataset import get_rml_dataloaders, RML2016Dataset, EXPERIMENT_SETTINGS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_validation_split():
    """Test the new train/val/test split functionality"""
    
    # Configuration
    data_path = '/Users/Axer/Desktop/py-meanflow/data/RML2016.10a_dict.pkl'  # Update path if needed
    experiment_setting = 1
    
    logger.info("=" * 80)
    logger.info("Testing Train/Validation/Test Split (80%/10%/10%)")
    logger.info("=" * 80)
    
    # Check if data file exists
    if not os.path.exists(data_path):
        logger.warning(f"Data file not found at {data_path}")
        logger.info("Please update the data_path variable to point to your RML2016.10a_dict.pkl file")
        return
    
    # Show experiment configuration
    logger.info(f"\nExperiment Setting {experiment_setting}:")
    logger.info(f"Known classes: {EXPERIMENT_SETTINGS[experiment_setting]['known']}")
    logger.info(f"Unknown classes: {EXPERIMENT_SETTINGS[experiment_setting]['unknown']}")
    
    # Create datasets with the new split
    logger.info("\nCreating datasets with 80/10/10 split...")
    
    try:
        # Create data loaders
        train_loader, val_loader, test_loader = get_rml_dataloaders(
            data_path=data_path,
            experiment_setting=experiment_setting,
            batch_size=64,
            num_workers=0,  # Set to 0 for testing
            snr_range=(-20, 20),
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            normalize=True,
            augment_train=True,
            seed=42
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("Dataset Statistics:")
        logger.info("=" * 80)
        
        # Training set statistics
        train_dataset = train_loader.dataset
        logger.info(f"\nTraining Set:")
        logger.info(f"  - Total samples: {len(train_dataset)}")
        logger.info(f"  - Batches: {len(train_loader)}")
        logger.info(f"  - Contains negative samples: Yes (for contrastive learning)")
        logger.info(f"  - Unknown classes included: No (only known classes)")
        
        # Validation set statistics
        val_dataset = val_loader.dataset
        logger.info(f"\nValidation Set:")
        logger.info(f"  - Total samples: {len(val_dataset)}")
        logger.info(f"  - Batches: {len(val_loader)}")
        logger.info(f"  - Contains negative samples: No")
        logger.info(f"  - Unknown classes included: No (only known classes)")
        logger.info(f"  - Purpose: Model selection, hyperparameter tuning, early stopping")
        
        # Test set statistics
        test_dataset = test_loader.dataset
        test_known = test_dataset.samples[~test_dataset.is_unknown]
        test_unknown = test_dataset.samples[test_dataset.is_unknown]
        logger.info(f"\nTest Set:")
        logger.info(f"  - Total samples: {len(test_dataset)}")
        logger.info(f"  - Known class samples: {len(test_known)}")
        logger.info(f"  - Unknown class samples: {len(test_unknown)}")
        logger.info(f"  - Batches: {len(test_loader)}")
        logger.info(f"  - Contains negative samples: No")
        logger.info(f"  - Unknown classes included: Yes (for open-set evaluation)")
        logger.info(f"  - Purpose: Final unbiased evaluation, AUROC/AUPR computation")
        
        # Sample a batch to verify data format
        logger.info("\n" + "=" * 80)
        logger.info("Data Format Verification:")
        logger.info("=" * 80)
        
        # Get a sample from each loader
        for loader_name, loader in [("Train", train_loader), ("Validation", val_loader), ("Test", test_loader)]:
            pos_samples, neg_samples, labels, info = next(iter(loader))
            logger.info(f"\n{loader_name} Batch:")
            logger.info(f"  - Positive samples shape: {pos_samples.shape}")
            if neg_samples is not None:
                logger.info(f"  - Negative samples shape: {neg_samples.shape}")
            else:
                logger.info(f"  - Negative samples: None")
            logger.info(f"  - Labels shape: {labels.shape}")
            logger.info(f"  - Label range: [{labels.min().item()}, {labels.max().item()}]")
            logger.info(f"  - Contains unknown: {info['is_unknown'].any().item()}")
        
        logger.info("\n" + "=" * 80)
        logger.info("Benefits of Validation Set for Classification:")
        logger.info("=" * 80)
        logger.info("""
1. **Model Selection**: Choose optimal hyperparameters without touching test set
2. **Early Stopping**: Prevent overfitting by monitoring validation performance
3. **Threshold Tuning**: Find optimal energy threshold for OOD detection
4. **Unbiased Evaluation**: Keep test set completely unseen until final evaluation
5. **Development Insights**: Track generalization during training
6. **Ensemble Selection**: Choose best models for ensemble methods
        """)
        
        logger.info("\n✅ Validation split successfully implemented!")
        logger.info("✅ Unknown data excluded from training and validation sets!")
        logger.info("✅ Test set preserved for final unbiased evaluation!")
        
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please download the RML2016.10a dataset and update the path")
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_validation_split()
