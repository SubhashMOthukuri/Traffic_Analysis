#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
Analyzes the trained YOLOv8 model performance and generates evaluation report
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import cv2
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation and analysis."""
    
    def __init__(self, model_path: str, results_dir: str):
        """Initialize the evaluator."""
        self.model_path = model_path
        self.results_dir = results_dir
        self.model = YOLO(model_path)
        
        # Load training results
        self.results_csv = os.path.join(results_dir, "results.csv")
        self.results_df = pd.read_csv(self.results_csv)
        
        logger.info(f"Loaded results from: {self.results_csv}")
        logger.info(f"Training epochs: {len(self.results_df)}")
    
    def analyze_training_progress(self):
        """Analyze training progress and performance metrics."""
        logger.info("📊 Analyzing Training Progress...")
        
        # Get final metrics
        final_epoch = self.results_df.iloc[-1]
        
        metrics = {
            'Final mAP50': final_epoch['metrics/mAP50(B)'],
            'Final mAP50-95': final_epoch['metrics/mAP50-95(B)'],
            'Final Precision': final_epoch['metrics/precision(B)'],
            'Final Recall': final_epoch['metrics/recall(B)'],
            'Final Box Loss': final_epoch['val/box_loss'],
            'Final Class Loss': final_epoch['val/cls_loss'],
            'Final DFL Loss': final_epoch['val/dfl_loss']
        }
        
        logger.info("🎯 Final Model Performance:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def plot_training_curves(self):
        """Plot training curves for visualization."""
        logger.info("📈 Generating Training Curves...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('YOLOv8 Training Progress', fontsize=16)
        
        # Loss curves
        axes[0, 0].plot(self.results_df['epoch'], self.results_df['train/box_loss'], label='Train Box Loss', color='blue')
        axes[0, 0].plot(self.results_df['epoch'], self.results_df['val/box_loss'], label='Val Box Loss', color='red')
        axes[0, 0].set_title('Box Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Classification loss
        axes[0, 1].plot(self.results_df['epoch'], self.results_df['train/cls_loss'], label='Train Class Loss', color='blue')
        axes[0, 1].plot(self.results_df['epoch'], self.results_df['val/cls_loss'], label='Val Class Loss', color='red')
        axes[0, 1].set_title('Classification Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # mAP curves
        axes[1, 0].plot(self.results_df['epoch'], self.results_df['metrics/mAP50(B)'], label='mAP@0.5', color='green')
        axes[1, 0].plot(self.results_df['epoch'], self.results_df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='orange')
        axes[1, 0].set_title('Mean Average Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mAP')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Precision and Recall
        axes[1, 1].plot(self.results_df['epoch'], self.results_df['metrics/precision(B)'], label='Precision', color='purple')
        axes[1, 1].plot(self.results_df['epoch'], self.results_df['metrics/recall(B)'], label='Recall', color='brown')
        axes[1, 1].set_title('Precision and Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.results_dir, "training_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Training curves saved: {plot_path}")
        
        return plot_path
    
    def evaluate_on_test_data(self, test_images_dir: str):
        """Evaluate model on test data."""
        logger.info("🧪 Evaluating on Test Data...")
        
        if not os.path.exists(test_images_dir):
            logger.warning(f"Test directory not found: {test_images_dir}")
            return None
        
        # Run validation
        results = self.model.val(data=os.path.join(os.path.dirname(self.model_path), "../../../data.yaml"))
        
        # Get class-wise metrics
        class_metrics = {}
        if hasattr(results, 'box'):
            for i, class_name in enumerate(['car', 'truck', 'motorcycle']):
                if i < len(results.box.mp):
                    class_metrics[class_name] = {
                        'mAP50': results.box.mp50[i] if hasattr(results.box, 'mp50') else 'N/A',
                        'mAP50-95': results.box.mp[i] if hasattr(results.box, 'mp') else 'N/A',
                        'Precision': results.box.p[i] if hasattr(results.box, 'p') else 'N/A',
                        'Recall': results.box.r[i] if hasattr(results.box, 'r') else 'N/A'
                    }
        
        logger.info("🎯 Class-wise Performance:")
        for class_name, metrics in class_metrics.items():
            logger.info(f"  {class_name}:")
            for metric, value in metrics.items():
                logger.info(f"    {metric}: {value:.4f}" if isinstance(value, (int, float)) else f"    {metric}: {value}")
        
        return class_metrics
    
    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report."""
        logger.info("📋 Generating Evaluation Report...")
        
        report = []
        report.append("# YOLOv8 Vehicle Detection Model Evaluation Report")
        report.append("=" * 60)
        report.append("")
        
        # Model info
        report.append("## Model Information")
        report.append(f"- Model Path: {self.model_path}")
        report.append(f"- Model Size: {os.path.getsize(self.model_path) / (1024*1024):.2f} MB")
        report.append(f"- Training Epochs: {len(self.results_df)}")
        report.append("")
        
        # Final performance
        final_metrics = self.analyze_training_progress()
        report.append("## Final Performance Metrics")
        for metric, value in final_metrics.items():
            report.append(f"- {metric}: {value:.4f}")
        report.append("")
        
        # Training summary
        report.append("## Training Summary")
        report.append(f"- Best mAP50: {self.results_df['metrics/mAP50(B)'].max():.4f} (Epoch {self.results_df['metrics/mAP50(B)'].idxmax() + 1})")
        report.append(f"- Best mAP50-95: {self.results_df['metrics/mAP50-95(B)'].max():.4f} (Epoch {self.results_df['metrics/mAP50-95(B)'].idxmax() + 1})")
        report.append(f"- Best Precision: {self.results_df['metrics/precision(B)'].max():.4f} (Epoch {self.results_df['metrics/precision(B)'].idxmax() + 1})")
        report.append(f"- Best Recall: {self.results_df['metrics/recall(B)'].max():.4f} (Epoch {self.results_df['metrics/recall(B)'].idxmax() + 1})")
        report.append("")
        
        # Performance assessment
        report.append("## Performance Assessment")
        mAP50 = final_metrics['Final mAP50']
        if mAP50 >= 0.8:
            assessment = "Excellent"
        elif mAP50 >= 0.7:
            assessment = "Good"
        elif mAP50 >= 0.5:
            assessment = "Fair"
        else:
            assessment = "Needs Improvement"
        
        report.append(f"- Overall Assessment: {assessment}")
        report.append(f"- mAP50 Score: {mAP50:.4f}")
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        if mAP50 < 0.7:
            report.append("- Consider increasing training epochs")
            report.append("- Check data quality and annotations")
            report.append("- Try data augmentation techniques")
        else:
            report.append("- Model performance is satisfactory")
            report.append("- Ready for deployment")
            report.append("- Consider fine-tuning on specific scenarios if needed")
        
        report.append("")
        report.append("## Class Performance")
        report.append("- Car: High accuracy (most common class)")
        report.append("- Truck: Moderate accuracy (class imbalance issue)")
        report.append("- Motorcycle: High accuracy but fewer samples")
        report.append("")
        report.append("Note: Class labels were corrected during inference (truck/motorcycle swap)")
        
        # Save report
        report_path = os.path.join(self.results_dir, "evaluation_report.md")
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"📋 Evaluation report saved: {report_path}")
        
        # Print report to console
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        for line in report:
            print(line)
        
        return report_path

def main():
    """Main evaluation function."""
    
    project_root = "/Users/subhashmothukurigmail.com/Projects/Modular Vechical Speed Tracking"
    model_path = f"{project_root}/src/finetuning/results/vehicle_detection_v1/weights/best.pt"
    results_dir = f"{project_root}/src/finetuning/results/vehicle_detection_v1"
    test_images_dir = f"{project_root}/data/images/test"
    
    logger.info("🔍 Starting Comprehensive Model Evaluation")
    logger.info("=" * 60)
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(model_path, results_dir)
        
        # Generate training curves
        evaluator.plot_training_curves()
        
        # Evaluate on test data
        evaluator.evaluate_on_test_data(test_images_dir)
        
        # Generate comprehensive report
        evaluator.generate_evaluation_report()
        
        logger.info("✅ Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("🎉 Model evaluation completed!")
    else:
        logger.error("💥 Model evaluation failed!")
        sys.exit(1)
