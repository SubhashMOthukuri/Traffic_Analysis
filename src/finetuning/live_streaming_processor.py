#!/usr/bin/env python3
"""
Production-Ready Live Video Streaming Processor
Optimized for real-time performance with multiple optimization strategies
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import logging
import time
import threading
import queue
from collections import deque
from typing import List, Dict, Tuple, Optional
import argparse
import torch

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ ONNX Runtime not available. Using PyTorch model.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiveStreamingProcessor:
    """Production-ready live video streaming processor with real-time optimizations."""
    
    def __init__(self, model_path: str, target_fps: int = 30, 
                 frame_skip: int = 1, input_size: int = 320, 
                 use_gpu: bool = True, async_processing: bool = True):
        """Initialize the live streaming processor with optimizations."""
        
        self.model_path = model_path
        self.target_fps = target_fps
        self.frame_skip = frame_skip
        self.input_size = input_size
        self.use_gpu = use_gpu
        self.async_processing = async_processing
        
        # Performance tracking
        self.frame_times = deque(maxlen=100)
        self.fps_history = deque(maxlen=100)
        self.processing_queue = queue.Queue(maxsize=10) if async_processing else None
        
        # Model initialization
        self.model = None
        self.onnx_session = None
        self.device = self._get_optimal_device()
        
        # Load models
        self._load_models()
        
        # Speed tracking
        self.track_history = {}
        self.speed_history = {}
        self.pixels_per_meter = 50
        self.time_per_frame = 1.0 / target_fps
        
        # Class names and colors (corrected mapping)
        self.class_names = {0: 'car', 1: 'motorcycle', 2: 'truck'}
        self.colors = {
            'car': (0, 255, 0),      # Green
            'truck': (255, 0, 0),   # Blue
            'motorcycle': (0, 0, 255) # Red
        }
        
        # ByteTracker for object tracking
        try:
            import supervision as sv
            self.byte_tracker = sv.ByteTrack()
        except ImportError:
            logger.warning("Supervision not available. Using basic tracking.")
            self.byte_tracker = None
        
        logger.info(f"🚀 Live Streaming Processor initialized")
        logger.info(f"   Target FPS: {target_fps}")
        logger.info(f"   Frame skip: {frame_skip}")
        logger.info(f"   Input size: {input_size}x{input_size}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Async processing: {async_processing}")

    def _get_optimal_device(self) -> str:
        """Determine the optimal device for inference."""
        if self.use_gpu:
            if torch.cuda.is_available():
                return 'cuda'
            elif torch.backends.mps.is_available():
                return 'mps'
        return 'cpu'

    def _load_models(self):
        """Load optimized models for live streaming."""
        onnx_path = self.model_path.replace('.pt', '.onnx')
        
        if ONNX_AVAILABLE and os.path.exists(onnx_path):
            logger.info(f"🔥 Loading ONNX model: {onnx_path}")
            # Configure ONNX providers for optimal performance
            providers = []
            if self.device == 'cuda':
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            elif self.device == 'mps':
                providers = ['CPUExecutionProvider']  # MPS not supported in ONNX
            else:
                providers = ['CPUExecutionProvider']
            
            self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
            logger.info("✅ ONNX model loaded successfully")
        
        # Always load PyTorch model for compatibility
        logger.info(f"📦 Loading PyTorch model: {self.model_path}")
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
        logger.info("✅ PyTorch model loaded successfully")

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Optimized frame preprocessing for live streaming."""
        # Resize to smaller input size for speed
        frame_resized = cv2.resize(frame, (self.input_size, self.input_size))
        return frame_resized

    def run_inference(self, frame: np.ndarray):
        """Run optimized inference."""
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Run inference with optimizations
        if self.onnx_session is not None:
            # Use PyTorch for now (ONNX post-processing is complex)
            results = self.model(processed_frame, device=self.device, 
                               conf=0.4, verbose=False)[0]
        else:
            results = self.model(processed_frame, device=self.device, 
                               conf=0.4, verbose=False)[0]
        
        return results

    def calculate_speed(self, track_id: int, current_position: Tuple[float, float]) -> float:
        """Calculate speed for tracked vehicle."""
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=10)
            self.speed_history[track_id] = deque(maxlen=5)
            return 0.0
        
        self.track_history[track_id].append(current_position)
        
        if len(self.track_history[track_id]) < 2:
            return 0.0
        
        # Calculate distance
        positions = list(self.track_history[track_id])
        total_distance = 0.0
        
        for i in range(1, len(positions)):
            x1, y1 = positions[i-1]
            x2, y2 = positions[i]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_distance += distance
        
        # Calculate speed
        time_elapsed = len(positions) * self.time_per_frame
        if time_elapsed == 0:
            return 0.0
        
        distance_meters = total_distance / self.pixels_per_meter
        speed_ms = distance_meters / time_elapsed
        speed_kmh = speed_ms * 3.6
        
        self.speed_history[track_id].append(speed_kmh)
        return np.mean(self.speed_history[track_id]) if self.speed_history[track_id] else 0.0

    def draw_detections(self, frame: np.ndarray, detections) -> np.ndarray:
        """Draw detections with optimized rendering."""
        annotated_frame = frame.copy()
        
        if hasattr(detections, 'boxes') and detections.boxes is not None:
            boxes = detections.boxes
            for i, box in enumerate(boxes):
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Scale back to original frame size
                scale_x = frame.shape[1] / self.input_size
                scale_y = frame.shape[0] / self.input_size
                
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                # Get class and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if confidence < 0.3:
                    continue
                
                class_name = self.class_names.get(class_id, 'unknown')
                color = self.colors.get(class_name, (255, 255, 255))
                
                # Calculate speed
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                speed = self.calculate_speed(i, (center_x, center_y))
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with speed
                label = f"{class_name}: {speed:.1f} km/h"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with optimizations."""
        start_time = time.time()
        
        # Run inference
        results = self.run_inference(frame)
        
        # Convert to supervision format for tracking
        if self.byte_tracker is not None:
            try:
                import supervision as sv
                detections = sv.Detections.from_ultralytics(results)
                detections = self.byte_tracker.update_with_detections(detections)
            except:
                detections = results
        else:
            detections = results
        
        # Draw detections
        annotated_frame = self.draw_detections(frame, detections)
        
        # Add performance info
        current_fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
        self.fps_history.append(current_fps)
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        # Draw performance metrics
        cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Target: {self.target_fps}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Skip: {self.frame_skip}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Size: {self.input_size}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame

    def process_live_stream(self, source: int = 0, output_path: str = None):
        """Process live video stream with real-time optimizations."""
        logger.info(f"🎥 Starting live stream processing from source: {source}")
        
        # Open video source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        
        # Set camera properties for optimal performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Reasonable resolution
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # Get actual properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        logger.info(f"📊 Stream properties: {width}x{height}, {fps} FPS")
        
        # Setup output video if specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info(f"📹 Recording to: {output_path}")
        
        # Processing variables
        frame_count = 0
        processed_count = 0
        start_time = time.time()
        
        logger.info("🚀 Starting live stream processing...")
        logger.info("Press 'q' to quit, 's' to save screenshot")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from stream")
                    break
                
                frame_count += 1
                
                # Skip frames for performance
                if frame_count % self.frame_skip != 0:
                    if out:
                        out.write(frame)
                    cv2.imshow('Live Stream', frame)
                    continue
                
                # Process frame
                try:
                    annotated_frame = self.process_frame(frame)
                    processed_count += 1
                    
                    # Save to output video
                    if out:
                        out.write(annotated_frame)
                    
                    # Display frame
                    cv2.imshow('Live Stream', annotated_frame)
                    
                    # Performance monitoring
                    if processed_count % 30 == 0:  # Every 30 processed frames
                        elapsed = time.time() - start_time
                        current_fps = processed_count / elapsed
                        logger.info(f"📊 Processed {processed_count} frames in {elapsed:.1f}s ({current_fps:.1f} FPS)")
                    
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {e}")
                    cv2.imshow('Live Stream', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested by user")
                    break
                elif key == ord('s'):
                    screenshot_path = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_path, annotated_frame)
                    logger.info(f"Screenshot saved: {screenshot_path}")
                
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            # Final statistics
            total_time = time.time() - start_time
            avg_fps = processed_count / total_time if total_time > 0 else 0
            
            logger.info(f"✅ Live stream processing completed!")
            logger.info(f"📊 Total frames: {frame_count}")
            logger.info(f"📊 Processed frames: {processed_count}")
            logger.info(f"📊 Average FPS: {avg_fps:.2f}")
            logger.info(f"📊 Target FPS: {self.target_fps}")
            logger.info(f"📊 Performance ratio: {avg_fps/self.target_fps:.2f}x")

    def process_video_file(self, input_path: str, output_path: str = None):
        """Process video file with live streaming optimizations."""
        logger.info(f"🎬 Processing video file: {input_path}")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"📊 Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup output video
        if output_path is None:
            input_path_obj = Path(input_path)
            output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_live_optimized.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing variables
        frame_count = 0
        processed_count = 0
        start_time = time.time()
        
        logger.info("🚀 Starting optimized video processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % self.frame_skip != 0:
                out.write(frame)
                continue
            
            # Process frame
            try:
                annotated_frame = self.process_frame(frame)
                out.write(annotated_frame)
                processed_count += 1
                
                # Progress update
                if processed_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    elapsed = time.time() - start_time
                    current_fps = processed_count / elapsed
                    logger.info(f"Progress: {progress:.1f}% - {current_fps:.1f} FPS")
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")
                out.write(frame)
        
        # Cleanup
        cap.release()
        out.release()
        
        # Final statistics
        total_time = time.time() - start_time
        avg_fps = processed_count / total_time if total_time > 0 else 0
        
        logger.info(f"✅ Video processing completed!")
        logger.info(f"📹 Output saved to: {output_path}")
        logger.info(f"📊 Average FPS: {avg_fps:.2f}")
        logger.info(f"📊 Performance ratio: {avg_fps/self.target_fps:.2f}x")

def main():
    """Main function for live streaming processor."""
    parser = argparse.ArgumentParser(description='Production-Ready Live Video Streaming Processor')
    parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam, file path for video)')
    parser.add_argument('--output', type=str, help='Output video path (optional)')
    parser.add_argument('--model', type=str, 
                       default='/Users/subhashmothukurigmail.com/Projects/Modular Vechical Speed Tracking/src/finetuning/results/vehicle_detection_v1/weights/best.pt',
                       help='Path to model file')
    parser.add_argument('--target-fps', type=int, default=30, help='Target FPS for processing')
    parser.add_argument('--frame-skip', type=int, default=1, help='Process every Nth frame')
    parser.add_argument('--input-size', type=int, default=320, help='Input image size for inference')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--no-async', action='store_true', help='Disable async processing')
    
    args = parser.parse_args()
    
    logger.info("🚀 Production-Ready Live Video Streaming Processor")
    logger.info("=" * 70)
    
    # Check if model exists
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return False
    
    try:
        # Initialize processor with optimizations
        processor = LiveStreamingProcessor(
            model_path=args.model,
            target_fps=args.target_fps,
            frame_skip=args.frame_skip,
            input_size=args.input_size,
            use_gpu=not args.no_gpu,
            async_processing=not args.no_async
        )
        
        # Determine if source is a file or camera
        if args.source.isdigit():
            # Process live stream (camera)
            processor.process_live_stream(int(args.source), args.output)
        elif os.path.exists(args.source):
            # Process video file
            processor.process_video_file(args.source, args.output)
        else:
            logger.error(f"Source not found: {args.source}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("🎉 Live streaming processing completed successfully!")
    else:
        logger.error("💥 Live streaming processing failed!")
        sys.exit(1)
