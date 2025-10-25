#!/usr/bin/env python3
"""
Test ONNX model performance on upview video
"""

import cv2
import numpy as np
import time
import os
import logging
from ultralytics import YOLO
import onnxruntime as ort

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_onnx_vs_pytorch():
    """Compare ONNX and PyTorch model performance on upview video."""
    
    # Paths
    pytorch_model_path = "/Users/subhashmothukurigmail.com/Projects/Modular Vechical Speed Tracking/src/finetuning/results/vehicle_detection_v1/weights/best.pt"
    onnx_model_path = "/Users/subhashmothukurigmail.com/Projects/Modular Vechical Speed Tracking/src/finetuning/results/vehicle_detection_v1/weights/best.onnx"
    video_path = "/Users/subhashmothukurigmail.com/Projects/Modular Vechical Speed Tracking/data/upview.mp4"
    
    # Check if files exist
    if not os.path.exists(pytorch_model_path):
        logger.error(f"PyTorch model not found: {pytorch_model_path}")
        return
    
    if not os.path.exists(onnx_model_path):
        logger.error(f"ONNX model not found: {onnx_model_path}")
        return
        
    if not os.path.exists(video_path):
        logger.error(f"Video not found: {video_path}")
        return
    
    logger.info("🚀 Testing ONNX vs PyTorch Model Performance")
    logger.info("=" * 60)
    
    # Load models
    logger.info("📦 Loading PyTorch model...")
    pytorch_model = YOLO(pytorch_model_path)
    
    logger.info("🔥 Loading ONNX model...")
    onnx_session = ort.InferenceSession(onnx_model_path)
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"📹 Video: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Test parameters
    test_frames = 50  # Test first 50 frames
    frame_count = 0
    
    # PyTorch timing
    logger.info("\n💻 Testing PyTorch Model...")
    pytorch_times = []
    pytorch_detections = []
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    
    while frame_count < test_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # PyTorch inference
        start_time = time.time()
        results = pytorch_model(frame, device='cpu', verbose=False)[0]
        end_time = time.time()
        
        pytorch_times.append(end_time - start_time)
        
        # Count detections
        if results.boxes is not None:
            pytorch_detections.append(len(results.boxes))
        else:
            pytorch_detections.append(0)
            
        frame_count += 1
        
        if frame_count % 10 == 0:
            logger.info(f"  Processed {frame_count}/{test_frames} frames")
    
    # ONNX timing
    logger.info("\n🔥 Testing ONNX Model...")
    onnx_times = []
    onnx_detections = []
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    frame_count = 0
    
    # Get ONNX input/output names
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    
    def preprocess_for_onnx(frame):
        """Preprocess frame for ONNX inference."""
        # Resize to model input size
        frame_resized = cv2.resize(frame, (640, 640))
        
        # Convert BGR to RGB and normalize
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW format
        frame_chw = frame_normalized.transpose(2, 0, 1)
        
        # Add batch dimension
        frame_batch = np.expand_dims(frame_chw, axis=0)
        
        return frame_batch
    
    while frame_count < test_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocess frame
        input_data = preprocess_for_onnx(frame)
        
        # ONNX inference
        start_time = time.time()
        outputs = onnx_session.run([output_name], {input_name: input_data})
        end_time = time.time()
        
        onnx_times.append(end_time - start_time)
        
        # Count detections (simplified - just count non-zero outputs)
        output = outputs[0]
        # This is a simplified detection count - in practice you'd need proper post-processing
        onnx_detections.append(np.sum(output[0, 4, :] > 0.3))  # confidence > 0.3
        
        frame_count += 1
        
        if frame_count % 10 == 0:
            logger.info(f"  Processed {frame_count}/{test_frames} frames")
    
    cap.release()
    
    # Calculate statistics
    pytorch_avg_time = np.mean(pytorch_times)
    pytorch_fps = 1.0 / pytorch_avg_time
    pytorch_std = np.std(pytorch_times)
    
    onnx_avg_time = np.mean(onnx_times)
    onnx_fps = 1.0 / onnx_avg_time
    onnx_std = np.std(onnx_times)
    
    speedup = pytorch_avg_time / onnx_avg_time
    
    # Results
    logger.info("\n📊 PERFORMANCE COMPARISON RESULTS")
    logger.info("=" * 60)
    logger.info(f"🎯 Tested {test_frames} frames from upview video")
    logger.info("")
    logger.info("💻 PyTorch Model:")
    logger.info(f"  Average time: {pytorch_avg_time*1000:.2f} ms")
    logger.info(f"  FPS: {pytorch_fps:.2f}")
    logger.info(f"  Std deviation: {pytorch_std*1000:.2f} ms")
    logger.info(f"  Avg detections: {np.mean(pytorch_detections):.1f}")
    logger.info("")
    logger.info("🔥 ONNX Model:")
    logger.info(f"  Average time: {onnx_avg_time*1000:.2f} ms")
    logger.info(f"  FPS: {onnx_fps:.2f}")
    logger.info(f"  Std deviation: {onnx_std*1000:.2f} ms")
    logger.info(f"  Avg detections: {np.mean(onnx_detections):.1f}")
    logger.info("")
    logger.info(f"⚡ Speedup: {speedup:.2f}x")
    logger.info(f"📈 FPS improvement: {onnx_fps - pytorch_fps:.2f}")
    
    if speedup > 1.0:
        logger.info("✅ ONNX model is FASTER!")
    else:
        logger.info("⚠️ PyTorch model is faster (ONNX overhead)")
    
    # Model size comparison
    pytorch_size = os.path.getsize(pytorch_model_path) / (1024*1024)
    onnx_size = os.path.getsize(onnx_model_path) / (1024*1024)
    
    logger.info("")
    logger.info("📦 MODEL SIZE COMPARISON:")
    logger.info(f"  PyTorch: {pytorch_size:.1f} MB")
    logger.info(f"  ONNX: {onnx_size:.1f} MB")
    logger.info(f"  Size ratio: {onnx_size/pytorch_size:.1f}x")

if __name__ == "__main__":
    test_onnx_vs_pytorch()
