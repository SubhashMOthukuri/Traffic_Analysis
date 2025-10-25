#!/usr/bin/env python3
"""
Accuracy Optimization Script
Implements data improvements and model tweaks to boost mAP/precision/recall
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import logging
from ultralytics import YOLO
import yaml
from typing import Dict, List, Tuple
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AccuracyOptimizer:
    """Accuracy optimization toolkit for YOLOv8."""
    
    def __init__(self, model_path: str, data_yaml: str):
        """Initialize the accuracy optimizer."""
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.model = YOLO(model_path)
        
        logger.info(f"Initialized accuracy optimizer with model: {model_path}")
    
    def enable_strong_augmentations(self, config: Dict = None):
        """Enable strong augmentations for better generalization."""
        logger.info("🔹 Enabling strong augmentations...")
        
        # Default strong augmentation config
        default_config = {
            # Geometric augmentations
            'degrees': 15.0,           # Rotation degrees
            'translate': 0.2,          # Translation fraction
            'scale': 0.9,              # Scale range
            'shear': 2.0,              # Shear degrees
            'perspective': 0.0001,     # Perspective distortion
            
            # Color augmentations
            'hsv_h': 0.015,           # HSV-Hue augmentation
            'hsv_s': 0.7,             # HSV-Saturation augmentation
            'hsv_v': 0.4,             # HSV-Value augmentation
            
            # Advanced augmentations
            'mosaic': 1.0,            # Mosaic probability
            'mixup': 0.15,            # Mixup probability
            'copy_paste': 0.3,        # Copy-paste probability
            
            # Other augmentations
            'flipud': 0.0,            # Vertical flip probability
            'fliplr': 0.5,            # Horizontal flip probability
            'erasing': 0.4,           # Random erasing probability
            'crop_fraction': 1.0      # Crop fraction
        }
        
        if config:
            default_config.update(config)
        
        logger.info("Strong augmentation parameters:")
        for key, value in default_config.items():
            logger.info(f"  {key}: {value}")
        
        return default_config
    
    def train_with_strong_augmentations(self, epochs: int = 50, imgsz: int = 640):
        """Train model with strong augmentations."""
        logger.info(f"🎯 Training with strong augmentations ({epochs} epochs)...")
        
        try:
            # Load model
            model = YOLO(self.model_path)
            
            # Get augmentation config
            aug_config = self.enable_strong_augmentations()
            
            # Train with strong augmentations
            results = model.train(
                data=self.data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=16,
                name='accuracy_optimized',
                patience=15,
                
                # Strong augmentations
                degrees=aug_config['degrees'],
                translate=aug_config['translate'],
                scale=aug_config['scale'],
                shear=aug_config['shear'],
                perspective=aug_config['perspective'],
                hsv_h=aug_config['hsv_h'],
                hsv_s=aug_config['hsv_s'],
                hsv_v=aug_config['hsv_v'],
                mosaic=aug_config['mosaic'],
                mixup=aug_config['mixup'],
                copy_paste=aug_config['copy_paste'],
                flipud=aug_config['flipud'],
                fliplr=aug_config['fliplr'],
                erasing=aug_config['erasing'],
                crop_fraction=aug_config['crop_fraction'],
                
                # Training parameters
                lr0=0.01,
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3,
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                
                # Loss weights
                box=7.5,
                cls=0.5,
                dfl=1.5
            )
            
            logger.info("✅ Training with strong augmentations completed!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return None
    
    def train_larger_model(self, model_size: str = 'yolov8s'):
        """Train with larger model (YOLOv8s or YOLOv8m) for better accuracy."""
        logger.info(f"🔹 Training with larger model: {model_size}")
        
        try:
            # Load larger model
            model = YOLO(f'{model_size}.pt')
            
            # Train with larger model
            results = model.train(
                data=self.data_yaml,
                epochs=100,
                imgsz=640,
                batch=16,
                name=f'{model_size}_optimized',
                patience=20,
                
                # Strong augmentations
                degrees=15.0,
                translate=0.2,
                scale=0.9,
                shear=2.0,
                perspective=0.0001,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                mosaic=1.0,
                mixup=0.15,
                copy_paste=0.3,
                
                # Training parameters
                lr0=0.01,
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3,
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                
                # Loss weights
                box=7.5,
                cls=0.5,
                dfl=1.5
            )
            
            logger.info(f"✅ Training with {model_size} completed!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Training with {model_size} failed: {e}")
            return None
    
    def train_higher_resolution(self, imgsz: int = 800):
        """Train with higher resolution for better small object detection."""
        logger.info(f"🔹 Training with higher resolution: {imgsz}x{imgsz}")
        
        try:
            # Load model
            model = YOLO(self.model_path)
            
            # Train with higher resolution
            results = model.train(
                data=self.data_yaml,
                epochs=100,
                imgsz=imgsz,
                batch=8,  # Smaller batch for higher resolution
                name=f'high_res_{imgsz}',
                patience=20,
                
                # Strong augmentations
                degrees=15.0,
                translate=0.2,
                scale=0.9,
                shear=2.0,
                perspective=0.0001,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                mosaic=1.0,
                mixup=0.15,
                copy_paste=0.3,
                
                # Training parameters
                lr0=0.01,
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3,
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                
                # Loss weights
                box=7.5,
                cls=0.5,
                dfl=1.5
            )
            
            logger.info(f"✅ Training with {imgsz}x{imgsz} resolution completed!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Training with higher resolution failed: {e}")
            return None
    
    def fine_tune_longer(self, additional_epochs: int = 30):
        """Fine-tune for longer with lower learning rate."""
        logger.info(f"🔹 Fine-tuning for {additional_epochs} additional epochs...")
        
        try:
            # Load best model
            model = YOLO(self.model_path)
            
            # Fine-tune with lower learning rate
            results = model.train(
                data=self.data_yaml,
                epochs=additional_epochs,
                imgsz=640,
                batch=16,
                name='fine_tuned_longer',
                patience=10,
                
                # Lower learning rate for fine-tuning
                lr0=0.001,  # Much lower LR
                lrf=0.001,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=2,
                warmup_momentum=0.8,
                warmup_bias_lr=0.01,
                
                # Moderate augmentations for fine-tuning
                degrees=10.0,
                translate=0.1,
                scale=0.95,
                shear=1.0,
                perspective=0.00005,
                hsv_h=0.01,
                hsv_s=0.5,
                hsv_v=0.3,
                mosaic=0.5,
                mixup=0.1,
                copy_paste=0.2,
                
                # Loss weights
                box=7.5,
                cls=0.5,
                dfl=1.5
            )
            
            logger.info(f"✅ Fine-tuning for {additional_epochs} epochs completed!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Fine-tuning failed: {e}")
            return None
    
    def analyze_class_balance(self):
        """Analyze class distribution in the dataset."""
        logger.info("🔹 Analyzing class balance...")
        
        try:
            # Load data config
            with open(self.data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
            
            # Count annotations in each split
            splits = ['train', 'val', 'test']
            class_counts = {}
            
            for split in splits:
                labels_dir = os.path.join(data_config['path'], 'labels', split)
                if os.path.exists(labels_dir):
                    split_counts = {0: 0, 1: 0, 2: 0}  # car, truck, motorcycle
                    
                    for label_file in os.listdir(labels_dir):
                        if label_file.endswith('.txt'):
                            with open(os.path.join(labels_dir, label_file), 'r') as f:
                                for line in f:
                                    if line.strip():
                                        class_id = int(line.split()[0])
                                        split_counts[class_id] += 1
                    
                    class_counts[split] = split_counts
            
            # Print analysis
            class_names = {0: 'car', 1: 'truck', 2: 'motorcycle'}
            
            logger.info("Class distribution analysis:")
            for split, counts in class_counts.items():
                logger.info(f"  {split}:")
                total = sum(counts.values())
                for class_id, count in counts.items():
                    percentage = (count / total) * 100 if total > 0 else 0
                    logger.info(f"    {class_names[class_id]}: {count} ({percentage:.1f}%)")
            
            return class_counts
            
        except Exception as e:
            logger.error(f"❌ Class balance analysis failed: {e}")
            return None

def main():
    """Main accuracy optimization function."""
    
    project_root = "/Users/subhashmothukurigmail.com/Projects/Modular Vechical Speed Tracking"
    model_path = f"{project_root}/src/finetuning/results/vehicle_detection_v1/weights/best.pt"
    data_yaml = f"{project_root}/data.yaml"
    
    logger.info("🎯 Starting Accuracy Optimization")
    logger.info("=" * 50)
    
    try:
        # Initialize optimizer
        optimizer = AccuracyOptimizer(model_path, data_yaml)
        
        # Analyze class balance
        optimizer.analyze_class_balance()
        
        # Run accuracy optimizations
        logger.info("Running accuracy optimizations...")
        
        # 1. Train with strong augmentations
        optimizer.train_with_strong_augmentations(epochs=50)
        
        # 2. Fine-tune longer
        optimizer.fine_tune_longer(additional_epochs=30)
        
        # 3. Train with higher resolution
        optimizer.train_higher_resolution(imgsz=800)
        
        # 4. Train larger model (optional)
        # optimizer.train_larger_model('yolov8s')
        
        logger.info("✅ All accuracy optimizations completed!")
        
    except Exception as e:
        logger.error(f"❌ Accuracy optimization failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("🎉 Accuracy optimization completed!")
    else:
        logger.error("💥 Accuracy optimization failed!")
        sys.exit(1)
