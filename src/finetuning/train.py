#!/usr/bin/env python3
"""
YOLOv8 Fine-tuning Script for Vehicle Speed Tracking
Run this script to start training your custom YOLOv8 model.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from finetuning import YOLOv8FineTuner
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('finetuning.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main training function."""
    logger.info("🚀 Starting YOLOv8 Fine-tuning for Vehicle Detection")
    
    # Project configuration
    project_root = "/Users/subhashmothukurigmail.com/Projects/Modular Vechical Speed Tracking"
    
    # Initialize fine-tuner
    fine_tuner = YOLOv8FineTuner(project_root)
    
    # Custom training configuration
    custom_config = {
        'pretrained_model': 'yolov8n.pt',  # Start with nano for faster training
        'epochs': 100,
        'imgsz': 640,
        'batch_size': 8,  # Reduced for better memory usage
        'patience': 20,
        'name': 'vehicle_detection_v1'
    }
    
    logger.info("📊 Training Configuration:")
    for key, value in custom_config.items():
        logger.info(f"  {key}: {value}")
    
    # Setup environment
    logger.info("🔧 Setting up training environment...")
    if not fine_tuner.setup_environment():
        logger.error("❌ Failed to setup environment")
        return False
    
    # Show dataset info
    logger.info("📁 Dataset Information:")
    logger.info("  Classes: car, truck, motorcycle")
    logger.info("  Train images: 160")
    logger.info("  Validation images: 30")
    logger.info("  Test images: 11")
    
    # Start training
    logger.info("🎯 Starting training process...")
    success, result = fine_tuner.train(custom_config)
    
    if success:
        logger.info("🎉 Training completed successfully!")
        logger.info(f"📁 Best model saved at: {result}")
        
        # Validate the trained model
        logger.info("🔍 Running model validation...")
        validation_results = fine_tuner.validate_model(result)
        
        if 'error' not in validation_results:
            logger.info("✅ Model validation completed!")
            logger.info(f"📈 Final Performance:")
            logger.info(f"  mAP50: {validation_results['mAP50']:.3f}")
            logger.info(f"  mAP50-95: {validation_results['mAP50_95']:.3f}")
            logger.info(f"  Precision: {validation_results['precision']:.3f}")
            logger.info(f"  Recall: {validation_results['recall']:.3f}")
        
        logger.info("🎯 Your custom YOLOv8 model is ready for vehicle detection!")
        return True
    else:
        logger.error(f"❌ Training failed: {result}")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            logger.info("🏁 Fine-tuning process completed successfully!")
        else:
            logger.error("💥 Fine-tuning process failed!")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        sys.exit(1)
