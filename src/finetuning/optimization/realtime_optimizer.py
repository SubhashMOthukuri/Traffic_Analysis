#!/usr/bin/env python3
"""
Real-Time Optimization Script
Implements speed and latency optimizations for 30-60 FPS live video feeds
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import logging
from ultralytics import YOLO
import torch
import time
import threading
import asyncio
from typing import Dict, List, Tuple, Optional
import onnxruntime as ort

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealTimeOptimizer:
    """Real-time optimization toolkit for YOLOv8."""
    
    def __init__(self, model_path: str):
        """Initialize the real-time optimizer."""
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.optimization_results = {}
        
        logger.info(f"Initialized real-time optimizer with model: {model_path}")
    
    def export_to_onnx(self, output_path: str = None, dynamic: bool = True, half: bool = True):
        """Export model to ONNX with optimizations."""
        logger.info("📦 Exporting model to ONNX with optimizations...")
        
        try:
            # Load model
            model = YOLO(self.model_path)
            
            # Export to ONNX with optimizations
            if output_path is None:
                output_path = self.model_path.replace('.pt', '_optimized.onnx')
            
            # Export with dynamic shapes and half precision
            exported_path = model.export(
                format='onnx',
                imgsz=640,
                dynamic=dynamic,
                half=half,
                simplify=True,
                opset=11
            )
            
            # Move to desired location
            if output_path != exported_path:
                shutil.move(exported_path, output_path)
            
            self.optimization_results['onnx_export'] = {
                'output_path': output_path,
                'dynamic': dynamic,
                'half_precision': half,
                'success': True
            }
            
            logger.info(f"✅ ONNX export completed!")
            logger.info(f"  Dynamic shapes: {dynamic}")
            logger.info(f"  Half precision: {half}")
            logger.info(f"  Saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ ONNX export failed: {e}")
            return None
    
    def benchmark_onnx_performance(self, onnx_path: str, iterations: int = 100):
        """Benchmark ONNX model performance."""
        logger.info(f"⚡ Benchmarking ONNX performance ({iterations} iterations)...")
        
        try:
            # Create ONNX Runtime session
            providers = ['CPUExecutionProvider']
            if torch.backends.mps.is_available():
                providers = ['CPUExecutionProvider']  # MPS not supported in ONNX Runtime yet
            
            session = ort.InferenceSession(onnx_path, providers=providers)
            
            # Get input details
            input_details = session.get_inputs()[0]
            input_shape = input_details.shape
            input_name = input_details.name
            
            # Create test input
            test_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
            
            # Warm up
            for _ in range(10):
                session.run(None, {input_name: test_input})
            
            # Benchmark
            times = []
            for i in range(iterations):
                start_time = time.time()
                session.run(None, {input_name: test_input})
                end_time = time.time()
                times.append(end_time - start_time)
                
                if (i + 1) % 20 == 0:
                    logger.info(f"  Completed {i + 1}/{iterations} iterations")
            
            # Calculate statistics
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            fps = 1.0 / avg_time
            
            self.optimization_results['onnx_benchmark'] = {
                'iterations': iterations,
                'avg_inference_time': avg_time,
                'std_inference_time': std_time,
                'min_inference_time': min_time,
                'max_inference_time': max_time,
                'fps': fps,
                'providers': providers
            }
            
            logger.info(f"✅ ONNX benchmark completed!")
            logger.info(f"  Average inference time: {avg_time:.4f}s")
            logger.info(f"  FPS: {fps:.2f}")
            logger.info(f"  Providers: {providers}")
            
            return self.optimization_results['onnx_benchmark']
            
        except Exception as e:
            logger.error(f"❌ ONNX benchmarking failed: {e}")
            return None
    
    def optimize_for_mac_mps(self):
        """Optimize for Mac MPS (Metal Performance Shaders)."""
        logger.info("🍎 Optimizing for Mac MPS...")
        
        try:
            if not torch.backends.mps.is_available():
                logger.warning("MPS not available on this system")
                return None
            
            # Load model
            model = YOLO(self.model_path)
            
            # Test MPS performance
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # CPU benchmark
            cpu_times = []
            for _ in range(50):
                start_time = time.time()
                model(test_image, device='cpu', verbose=False)
                end_time = time.time()
                cpu_times.append(end_time - start_time)
            
            # MPS benchmark
            mps_times = []
            for _ in range(50):
                start_time = time.time()
                model(test_image, device='mps', verbose=False)
                end_time = time.time()
                mps_times.append(end_time - start_time)
            
            cpu_avg = np.mean(cpu_times)
            mps_avg = np.mean(mps_times)
            speedup = cpu_avg / mps_avg
            
            self.optimization_results['mps_optimization'] = {
                'cpu_avg_time': cpu_avg,
                'mps_avg_time': mps_avg,
                'speedup': speedup,
                'mps_available': torch.backends.mps.is_available()
            }
            
            logger.info(f"✅ MPS optimization completed!")
            logger.info(f"  CPU average time: {cpu_avg:.4f}s")
            logger.info(f"  MPS average time: {mps_avg:.4f}s")
            logger.info(f"  Speedup: {speedup:.2f}x")
            
            return self.optimization_results['mps_optimization']
            
        except Exception as e:
            logger.error(f"❌ MPS optimization failed: {e}")
            return None
    
    def create_optimized_video_processor(self, skip_frames: int = 1, resize_factor: float = 1.0):
        """Create optimized video processor with frame skipping and resizing."""
        logger.info(f"🎬 Creating optimized video processor (skip={skip_frames}, resize={resize_factor})...")
        
        try:
            # Load model
            model = YOLO(self.model_path)
            
            # Optimized processing function
            def process_frame_optimized(frame, frame_count):
                # Skip frames
                if frame_count % (skip_frames + 1) != 0:
                    return None
                
                # Resize frame for faster processing
                if resize_factor != 1.0:
                    h, w = frame.shape[:2]
                    new_h, new_w = int(h * resize_factor), int(w * resize_factor)
                    frame = cv2.resize(frame, (new_w, new_h))
                
                # Run inference
                results = model(frame, device='mps' if torch.backends.mps.is_available() else 'cpu', verbose=False)
                
                return results
            
            self.optimization_results['optimized_processor'] = {
                'skip_frames': skip_frames,
                'resize_factor': resize_factor,
                'device': 'mps' if torch.backends.mps.is_available() else 'cpu'
            }
            
            logger.info(f"✅ Optimized video processor created!")
            logger.info(f"  Skip frames: {skip_frames}")
            logger.info(f"  Resize factor: {resize_factor}")
            logger.info(f"  Device: {self.optimization_results['optimized_processor']['device']}")
            
            return process_frame_optimized
            
        except Exception as e:
            logger.error(f"❌ Optimized processor creation failed: {e}")
            return None
    
    def benchmark_async_processing(self, video_path: str, max_frames: int = 100):
        """Benchmark async processing for real-time performance."""
        logger.info(f"🔄 Benchmarking async processing...")
        
        try:
            # Load model
            model = YOLO(self.model_path)
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Async processing function
            async def async_inference(frame):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, model, frame)
            
            # Benchmark async processing
            frame_count = 0
            start_time = time.time()
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize for faster processing
                frame = cv2.resize(frame, (640, 640))
                
                # Run async inference
                asyncio.run(async_inference(frame))
                
                frame_count += 1
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_fps = frame_count / total_time
            
            cap.release()
            
            self.optimization_results['async_benchmark'] = {
                'frames_processed': frame_count,
                'total_time': total_time,
                'avg_fps': avg_fps,
                'target_fps': fps
            }
            
            logger.info(f"✅ Async processing benchmark completed!")
            logger.info(f"  Frames processed: {frame_count}")
            logger.info(f"  Average FPS: {avg_fps:.2f}")
            logger.info(f"  Target FPS: {fps}")
            
            return self.optimization_results['async_benchmark']
            
        except Exception as e:
            logger.error(f"❌ Async processing benchmark failed: {e}")
            return None
    
    def create_production_config(self, output_path: str = None):
        """Create production configuration for real-time deployment."""
        logger.info("🏭 Creating production configuration...")
        
        try:
            if output_path is None:
                output_path = "production_config.yaml"
            
            # Production configuration
            config = {
                'model': {
                    'path': self.model_path,
                    'format': 'onnx',
                    'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
                    'batch_size': 1,
                    'imgsz': 640
                },
                'optimization': {
                    'skip_frames': 1,
                    'resize_factor': 1.0,
                    'async_processing': True,
                    'half_precision': True,
                    'dynamic_shapes': True
                },
                'performance': {
                    'target_fps': 30,
                    'max_latency_ms': 33,
                    'memory_limit_mb': 1024
                },
                'augmentation': {
                    'enabled': False,  # Disable for inference
                    'mosaic': 0.0,
                    'mixup': 0.0
                }
            }
            
            # Save configuration
            import yaml
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            self.optimization_results['production_config'] = {
                'output_path': output_path,
                'config': config
            }
            
            logger.info(f"✅ Production configuration created!")
            logger.info(f"  Saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Production configuration creation failed: {e}")
            return None
    
    def generate_optimization_report(self, output_path: str = None):
        """Generate real-time optimization report."""
        logger.info("📋 Generating real-time optimization report...")
        
        if output_path is None:
            output_path = "realtime_optimization_report.json"
        
        # Save results
        import json
        with open(output_path, 'w') as f:
            json.dump(self.optimization_results, f, indent=2, default=str)
        
        logger.info(f"📋 Optimization report saved: {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("REAL-TIME OPTIMIZATION REPORT")
        print("="*60)
        
        for optimization_type, results in self.optimization_results.items():
            print(f"\n⚡ {optimization_type.upper()}:")
            for key, value in results.items():
                print(f"  {key}: {value}")
        
        return output_path

def main():
    """Main real-time optimization function."""
    
    project_root = "/Users/subhashmothukurigmail.com/Projects/Modular Vechical Speed Tracking"
    model_path = f"{project_root}/src/finetuning/results/vehicle_detection_v1/weights/best.pt"
    test_video = f"{project_root}/data/upview.mp4"
    
    logger.info("⚡ Starting Real-Time Optimization")
    logger.info("=" * 50)
    
    try:
        # Initialize optimizer
        optimizer = RealTimeOptimizer(model_path)
        
        # Run real-time optimizations
        logger.info("Running real-time optimizations...")
        
        # 1. Export to ONNX
        onnx_path = optimizer.export_to_onnx()
        
        # 2. Benchmark ONNX performance
        if onnx_path:
            optimizer.benchmark_onnx_performance(onnx_path, iterations=50)
        
        # 3. Optimize for Mac MPS
        optimizer.optimize_for_mac_mps()
        
        # 4. Create optimized video processor
        optimizer.create_optimized_video_processor(skip_frames=1, resize_factor=1.0)
        
        # 5. Benchmark async processing
        if os.path.exists(test_video):
            optimizer.benchmark_async_processing(test_video, max_frames=50)
        
        # 6. Create production configuration
        optimizer.create_production_config()
        
        # 7. Generate report
        optimizer.generate_optimization_report()
        
        logger.info("✅ All real-time optimizations completed!")
        
    except Exception as e:
        logger.error(f"❌ Real-time optimization failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("🎉 Real-time optimization completed!")
    else:
        logger.error("💥 Real-time optimization failed!")
        sys.exit(1)
