"""
YOLOv8 Fine-tuning Module for Vehicle Speed Tracking
This module provides a clean interface for training custom YOLOv8 models.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOv8FineTuner:
    """YOLOv8 Fine-tuning class for vehicle detection."""
    
    def __init__(self, project_root: str):
        """Initialize the fine-tuner with project root path."""
        self.project_root = Path(project_root)
        self.data_yaml = self.project_root / "data.yaml"
        self.models_dir = self.project_root / "src" / "finetuning" / "models"
        self.results_dir = self.project_root / "src" / "finetuning" / "results"
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        self.config = {
            'pretrained_model': 'yolov8n.pt',
            'epochs': 100,
            'imgsz': 640,
            'batch_size': 16,
            'device': 'auto',
            'patience': 20,
            'save_period': 10,
            'project': str(self.results_dir),
            'name': 'vehicle_detection'
        }
        
        logger.info(f"YOLOv8 Fine-tuner initialized")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Results directory: {self.results_dir}")
    
    def setup_environment(self) -> bool:
        """Setup the training environment and download pretrained weights."""
        try:
            logger.info("Setting up fine-tuning environment...")
            
            # Check if data.yaml exists
            if not self.data_yaml.exists():
                logger.error(f"Data configuration file not found: {self.data_yaml}")
                return False
            
            # Check device availability
            if torch.cuda.is_available():
                self.config['device'] = 'cuda'
                logger.info(f"CUDA available! Using GPU: {torch.cuda.get_device_name()}")
            else:
                self.config['device'] = 'cpu'
                logger.info("CUDA not available, using CPU")
            
            # Download pretrained model
            logger.info(f"Downloading pretrained model: {self.config['pretrained_model']}")
            model = YOLO(self.config['pretrained_model'])
            
            # Save model info
            model_info = {
                'model_name': self.config['pretrained_model'],
                'device': self.config['device'],
                'data_yaml': str(self.data_yaml),
                'project_root': str(self.project_root)
            }
            
            logger.info("Environment setup completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            return False
    
    def train(self, custom_config: Optional[Dict] = None) -> Tuple[bool, str]:
        """Train the YOLOv8 model with custom configuration."""
        try:
            # Update config with custom settings
            if custom_config:
                self.config.update(custom_config)
            
            logger.info("Starting YOLOv8 training...")
            logger.info(f"Configuration: {self.config}")
            
            # Load pretrained model
            model = YOLO(self.config['pretrained_model'])
            
            # Start training
            results = model.train(
                data=str(self.data_yaml),
                epochs=self.config['epochs'],
                imgsz=self.config['imgsz'],
                batch=self.config['batch_size'],
                device=self.config['device'],
                patience=self.config['patience'],
                save_period=self.config['save_period'],
                project=self.config['project'],
                name=self.config['name'],
                # Transfer learning settings
                pretrained=True,
                freeze=None,
                # Data augmentation
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=0.0,
                translate=0.1,
                scale=0.5,
                shear=0.0,
                perspective=0.0,
                flipud=0.0,
                fliplr=0.5,
                mosaic=1.0,
                mixup=0.0,
                copy_paste=0.0,
                # Optimization
                optimizer='AdamW',
                lr0=0.01,
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3,
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                # Loss function
                box=7.5,
                cls=0.5,
                dfl=1.5,
                # Validation
                val=True,
                plots=True,
                save=True,
                verbose=True,
            )
            
            # Get the best model path
            best_model_path = results.save_dir / "weights" / "best.pt"
            
            logger.info("Training completed successfully!")
            logger.info(f"Best model saved at: {best_model_path}")
            
            # Run validation
            logger.info("Running final validation...")
            val_results = model.val()
            
            logger.info(f"Final Validation Results:")
            logger.info(f"  mAP50: {val_results.box.map50:.3f}")
            logger.info(f"  mAP50-95: {val_results.box.map:.3f}")
            
            return True, str(best_model_path)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False, str(e)
    
    def validate_model(self, model_path: str) -> Dict:
        """Validate a trained model."""
        try:
            logger.info(f"Validating model: {model_path}")
            model = YOLO(model_path)
            results = model.val()
            
            validation_results = {
                'mAP50': float(results.box.map50),
                'mAP50_95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'model_path': model_path
            }
            
            logger.info("Validation completed:")
            for key, value in validation_results.items():
                if key != 'model_path':
                    logger.info(f"  {key}: {value:.3f}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {'error': str(e)}
    
    def get_available_models(self) -> Dict[str, str]:
        """Get available pretrained YOLOv8 models."""
        return {
            'yolov8n.pt': 'Nano - Fastest, smallest (6MB)',
            'yolov8s.pt': 'Small - Good balance (22MB)',
            'yolov8m.pt': 'Medium - Better accuracy (50MB)',
            'yolov8l.pt': 'Large - High accuracy (87MB)',
            'yolov8x.pt': 'Extra Large - Best accuracy (136MB)'
        }

def main():
    """Main function to run fine-tuning."""
    project_root = "/Users/subhashmothukurigmail.com/Projects/Modular Vechical Speed Tracking"
    
    # Initialize fine-tuner
    fine_tuner = YOLOv8FineTuner(project_root)
    
    # Setup environment
    if not fine_tuner.setup_environment():
        logger.error("Failed to setup environment")
        return False
    
    # Show available models
    logger.info("Available pretrained models:")
    for model, description in fine_tuner.get_available_models().items():
        logger.info(f"  {model}: {description}")
    
    # Start training
    success, result = fine_tuner.train()
    
    if success:
        logger.info("🎉 Training completed successfully!")
        logger.info(f"Best model: {result}")
        
        # Validate the trained model
        validation_results = fine_tuner.validate_model(result)
        if 'error' not in validation_results:
            logger.info("✅ Model validation completed!")
        
        return True
    else:
        logger.error(f"❌ Training failed: {result}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
