#!/usr/bin/env python3
"""
YOLOv8 Fine-tuning Script for Vehicle Speed Tracking
This script trains a YOLOv8 model using transfer learning on your custom dataset.
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main training function with transfer learning."""
    
    # Configuration
    CONFIG = {
        'data_yaml': '/Users/subhashmothukurigmail.com/Projects/Modular Vechical Speed Tracking/data.yaml',
        'pretrained_model': 'yolov8n.pt',  # Start with nano for faster training
        'epochs': 100,
        'imgsz': 640,
        'batch_size': 16,
        'device': 'cpu',  # Change to 'cuda' if you have GPU
        'patience': 20,  # Early stopping patience
        'save_period': 10,  # Save checkpoint every 10 epochs
        'project': 'vehicle_detection',
        'name': 'yolov8n_vehicle_tracking'
    }
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        CONFIG['device'] = 'cuda'
        logger.info(f"CUDA available! Using GPU: {torch.cuda.get_device_name()}")
    else:
        logger.info("CUDA not available, using CPU")
    
    logger.info("Starting YOLOv8 fine-tuning with transfer learning...")
    logger.info(f"Configuration: {CONFIG}")
    
    # Load pretrained model
    logger.info(f"Loading pretrained model: {CONFIG['pretrained_model']}")
    model = YOLO(CONFIG['pretrained_model'])
    
    # Verify data configuration
    if not os.path.exists(CONFIG['data_yaml']):
        logger.error(f"Data YAML file not found: {CONFIG['data_yaml']}")
        return False
    
    logger.info(f"Using data configuration: {CONFIG['data_yaml']}")
    
    # Start training with transfer learning
    logger.info("Starting training...")
    try:
        results = model.train(
            data=CONFIG['data_yaml'],
            epochs=CONFIG['epochs'],
            imgsz=CONFIG['imgsz'],
            batch=CONFIG['batch_size'],
            device=CONFIG['device'],
            patience=CONFIG['patience'],
            save_period=CONFIG['save_period'],
            project=CONFIG['project'],
            name=CONFIG['name'],
            # Transfer learning settings
            pretrained=True,  # Use pretrained weights
            freeze=None,  # Don't freeze any layers
            # Data augmentation
            hsv_h=0.015,  # HSV-Hue augmentation
            hsv_s=0.7,    # HSV-Saturation augmentation
            hsv_v=0.4,    # HSV-Value augmentation
            degrees=0.0,  # Rotation degrees
            translate=0.1, # Translation fraction
            scale=0.5,    # Scale gain
            shear=0.0,   # Shear degrees
            perspective=0.0, # Perspective gain
            flipud=0.0,  # Flip up-down probability
            fliplr=0.5,  # Flip left-right probability
            mosaic=1.0,  # Mosaic probability
            mixup=0.0,   # Mixup probability
            copy_paste=0.0, # Copy-paste probability
            # Optimization
            optimizer='AdamW',  # Optimizer
            lr0=0.01,    # Initial learning rate
            lrf=0.01,    # Final learning rate
            momentum=0.937,  # Momentum
            weight_decay=0.0005,  # Weight decay
            warmup_epochs=3,  # Warmup epochs
            warmup_momentum=0.8,  # Warmup momentum
            warmup_bias_lr=0.1,   # Warmup bias learning rate
            # Loss function
            box=7.5,     # Box loss gain
            cls=0.5,     # Classification loss gain
            dfl=1.5,     # DFL loss gain
            # Validation
            val=True,    # Validate during training
            plots=True,  # Generate plots
            save=True,   # Save checkpoints
            save_txt=False,  # Save results to txt
            save_conf=False, # Save confidences
            save_crop=False, # Save cropped images
            show_labels=True, # Show labels
            show_conf=True,   # Show confidences
            verbose=True,    # Verbose output
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Best model saved at: {results.save_dir}")
        
        # Validate the trained model
        logger.info("Running validation...")
        val_results = model.val()
        logger.info(f"Validation mAP50: {val_results.box.map50:.3f}")
        logger.info(f"Validation mAP50-95: {val_results.box.map:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

def download_pretrained_weights():
    """Download YOLOv8 pretrained weights."""
    logger.info("Downloading YOLOv8 pretrained weights...")
    
    # Available models (choose based on your needs)
    models = {
        'yolov8n.pt': 'Nano - Fastest, smallest',
        'yolov8s.pt': 'Small - Good balance',
        'yolov8m.pt': 'Medium - Better accuracy',
        'yolov8l.pt': 'Large - High accuracy',
        'yolov8x.pt': 'Extra Large - Best accuracy'
    }
    
    logger.info("Available pretrained models:")
    for model, description in models.items():
        logger.info(f"  {model}: {description}")
    
    # Download nano model (recommended for starting)
    model = YOLO('yolov8n.pt')
    logger.info("YOLOv8n pretrained weights downloaded successfully!")
    
    return True

if __name__ == "__main__":
    # Download pretrained weights first
    if not download_pretrained_weights():
        logger.error("Failed to download pretrained weights")
        sys.exit(1)
    
    # Start training
    success = main()
    if success:
        logger.info("🎉 Training completed successfully!")
        logger.info("Your custom YOLOv8 model is ready for vehicle detection!")
    else:
        logger.error("❌ Training failed!")
        sys.exit(1)
