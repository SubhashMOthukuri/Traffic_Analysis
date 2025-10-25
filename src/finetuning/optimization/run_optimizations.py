#!/usr/bin/env python3
"""
Comprehensive Optimization Runner
Runs all optimization strategies for maximum performance
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_accuracy_optimization():
    """Run accuracy optimization."""
    logger.info("🎯 Running Accuracy Optimization...")
    
    try:
        from accuracy_optimizer import AccuracyOptimizer
        
        project_root = "/Users/subhashmothukurigmail.com/Projects/Modular Vechical Speed Tracking"
        model_path = f"{project_root}/src/finetuning/results/vehicle_detection_v1/weights/best.pt"
        data_yaml = f"{project_root}/data.yaml"
        
        optimizer = AccuracyOptimizer(model_path, data_yaml)
        
        # Analyze class balance
        optimizer.analyze_class_balance()
        
        # Train with strong augmentations
        optimizer.train_with_strong_augmentations(epochs=30)
        
        logger.info("✅ Accuracy optimization completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Accuracy optimization failed: {e}")
        return False

def run_realtime_optimization():
    """Run real-time optimization."""
    logger.info("⚡ Running Real-Time Optimization...")
    
    try:
        from realtime_optimizer import RealTimeOptimizer
        
        project_root = "/Users/subhashmothukurigmail.com/Projects/Modular Vechical Speed Tracking"
        model_path = f"{project_root}/src/finetuning/results/vehicle_detection_v1/weights/best.pt"
        
        optimizer = RealTimeOptimizer(model_path)
        
        # Export to ONNX
        onnx_path = optimizer.export_to_onnx()
        
        # Benchmark ONNX performance
        if onnx_path:
            optimizer.benchmark_onnx_performance(onnx_path, iterations=30)
        
        # Optimize for Mac MPS
        optimizer.optimize_for_mac_mps()
        
        # Create production configuration
        optimizer.create_production_config()
        
        logger.info("✅ Real-time optimization completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Real-time optimization failed: {e}")
        return False

def main():
    """Main optimization runner."""
    
    parser = argparse.ArgumentParser(description='Run model optimizations')
    parser.add_argument('--accuracy', action='store_true', help='Run accuracy optimization')
    parser.add_argument('--realtime', action='store_true', help='Run real-time optimization')
    parser.add_argument('--all', action='store_true', help='Run all optimizations')
    
    args = parser.parse_args()
    
    logger.info("🔧 Starting Comprehensive Model Optimization")
    logger.info("=" * 60)
    
    success = True
    
    if args.all or args.accuracy:
        success &= run_accuracy_optimization()
    
    if args.all or args.realtime:
        success &= run_realtime_optimization()
    
    if not args.accuracy and not args.realtime and not args.all:
        logger.info("No optimization type specified. Use --help for options.")
        logger.info("Available options:")
        logger.info("  --accuracy   : Run accuracy optimization")
        logger.info("  --realtime   : Run real-time optimization")
        logger.info("  --all        : Run all optimizations")
        return False
    
    if success:
        logger.info("🎉 All optimizations completed successfully!")
    else:
        logger.error("💥 Some optimizations failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("🏆 Optimization process completed!")
    else:
        logger.error("💥 Optimization process failed!")
        sys.exit(1)
