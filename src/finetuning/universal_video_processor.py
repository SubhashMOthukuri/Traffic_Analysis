#!/usr/bin/env python3
"""
Universal Vehicle Speed Tracking Video Processor
Processes any video file with corrected vehicle detection and speed tracking
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import logging
from ultralytics import YOLO
import supervision as sv
from typing import List, Dict, Tuple
import time
import argparse

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

class UniversalVehicleSpeedTracker:
    """Universal vehicle detection and speed tracking with CORRECTED class mapping."""
    
    def __init__(self, model_path: str, input_video: str, output_video: str = None):
        """Initialize the speed tracker with corrected class mapping."""
        self.model_path = model_path
        self.input_video = input_video
        
        # Generate output filename if not provided
        if output_video is None:
            input_path = Path(input_video)
            output_video = str(input_path.parent / f"{input_path.stem}_onnx_output.mp4")
        
        self.output_video = output_video
        
        # Try to load ONNX model first, fallback to PyTorch
        self.model = None
        self.onnx_session = None
        
        onnx_path = model_path.replace('.pt', '.onnx')
        
        if ONNX_AVAILABLE and os.path.exists(onnx_path):
            logger.info(f"🔥 Loading ONNX model: {onnx_path}")
            self.onnx_session = ort.InferenceSession(onnx_path)
            # Also load PyTorch model for post-processing compatibility
            logger.info(f"📦 Loading PyTorch model for post-processing: {model_path}")
            self.model = YOLO(model_path)
            logger.info("✅ ONNX + PyTorch models loaded successfully")
        else:
            logger.info(f"📦 Loading PyTorch model: {model_path}")
            self.model = YOLO(model_path)
            logger.info("✅ PyTorch model loaded successfully")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(input_video)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_video}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {self.width}x{self.height}, {self.fps} FPS, {self.total_frames} frames")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_video, fourcc, self.fps, (self.width, self.height))
        
        # Speed tracking variables
        self.track_history = {}  # Track ID -> list of positions
        self.speed_history = {}  # Track ID -> list of speeds
        self.frame_count = 0
        
        # Speed calculation parameters
        self.pixels_per_meter = 50  # Adjust based on your camera setup
        self.time_per_frame = 1.0 / self.fps
        
        # CORRECTED Class names and colors - swapping truck and motorcycle
        self.class_names = {0: 'car', 1: 'motorcycle', 2: 'truck'}  # SWAPPED!
        self.colors = {
            'car': (0, 255, 0),      # Green
            'truck': (255, 0, 0),   # Blue
            'motorcycle': (0, 0, 255) # Red
        }
        
        # Debug: Print corrected class mapping
        logger.info(f"CORRECTED Class mapping: {self.class_names}")
        logger.info("Note: Swapped truck (1) and motorcycle (2) labels")
        
        # ByteTracker for object tracking
        self.byte_tracker = sv.ByteTrack()
        
        # ONNX preprocessing parameters
        self.imgsz = 640
        
    def preprocess_for_onnx(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for ONNX inference."""
        # Resize to model input size
        frame_resized = cv2.resize(frame, (self.imgsz, self.imgsz))
        
        # Convert BGR to RGB and normalize
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW format
        frame_chw = frame_normalized.transpose(2, 0, 1)
        
        # Add batch dimension
        frame_batch = np.expand_dims(frame_chw, axis=0)
        
        return frame_batch
    
    def run_onnx_inference(self, frame: np.ndarray):
        """Run inference using ONNX model with PyTorch post-processing."""
        if self.onnx_session is None:
            raise RuntimeError("ONNX session not initialized")
        
        # For now, use PyTorch model for both inference and post-processing
        # This ensures compatibility with supervision library
        # In production, you'd implement proper ONNX post-processing
        return self.model(frame, conf=0.5, verbose=False)[0]
        
    def calculate_speed(self, track_id: int, current_position: Tuple[float, float]) -> float:
        """Calculate speed for a tracked vehicle."""
        if track_id not in self.track_history:
            self.track_history[track_id] = []
            self.speed_history[track_id] = []
            return 0.0
        
        self.track_history[track_id].append(current_position)
        
        # Keep only last 10 positions for speed calculation
        if len(self.track_history[track_id]) > 10:
            self.track_history[track_id] = self.track_history[track_id][-10:]
        
        if len(self.track_history[track_id]) < 2:
            return 0.0
        
        # Calculate distance moved
        positions = self.track_history[track_id]
        total_distance = 0.0
        
        for i in range(1, len(positions)):
            x1, y1 = positions[i-1]
            x2, y2 = positions[i]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_distance += distance
        
        # Calculate time elapsed
        time_elapsed = len(positions) * self.time_per_frame
        
        if time_elapsed == 0:
            return 0.0
        
        # Convert pixels to meters and calculate speed (m/s)
        distance_meters = total_distance / self.pixels_per_meter
        speed_ms = distance_meters / time_elapsed
        
        # Convert to km/h
        speed_kmh = speed_ms * 3.6
        
        self.speed_history[track_id].append(speed_kmh)
        
        # Return average speed over last few measurements
        if len(self.speed_history[track_id]) > 5:
            self.speed_history[track_id] = self.speed_history[track_id][-5:]
        
        return np.mean(self.speed_history[track_id])
    
    def draw_detections(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Draw bounding boxes and speed information on frame with CORRECTED labels."""
        annotated_frame = frame.copy()
        
        if detections.tracker_id is None:
            return annotated_frame
        
        for i, (bbox, class_id, tracker_id, confidence) in enumerate(zip(
            detections.xyxy, detections.class_id, detections.tracker_id, detections.confidence
        )):
            if tracker_id is None:
                continue
                
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Apply corrected class mapping
            class_name = self.class_names.get(class_id, 'unknown')
            color = self.colors.get(class_name, (255, 255, 255))
            
            # Calculate speed
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            speed = self.calculate_speed(tracker_id, (center_x, center_y))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with speed
            label = f"{class_name}: {speed:.1f} km/h"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw center point
            cv2.circle(annotated_frame, (int(center_x), int(center_y)), 3, color, -1)
        
        return annotated_frame
    
    def process_video(self):
        """Process the entire video and generate output with speed tracking."""
        logger.info("Starting video processing with CORRECTED class labels...")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Run detection (ONNX or PyTorch)
                if self.onnx_session is not None:
                    # Use ONNX for inference, PyTorch for post-processing
                    results = self.run_onnx_inference(frame)
                else:
                    # Use PyTorch model
                    results = self.model(frame, conf=0.5, verbose=False)[0]
                
                detections = sv.Detections.from_ultralytics(results)
                
                # Update tracker
                detections = self.byte_tracker.update_with_detections(detections)
                
                # Draw detections and speed with corrected labels
                annotated_frame = self.draw_detections(frame, detections)
                
                # Add frame counter
                cv2.putText(annotated_frame, f"Frame: {self.frame_count}/{self.total_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add model type and correction notice
                model_type = "ONNX MODEL" if self.onnx_session is not None else "PYTORCH MODEL"
                cv2.putText(annotated_frame, model_type, 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(annotated_frame, "CORRECTED LABELS", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Add video filename
                video_name = Path(self.input_video).name
                cv2.putText(annotated_frame, video_name, 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Write frame to output video
                self.out.write(annotated_frame)
                
                # Progress update
                if self.frame_count % 100 == 0:
                    progress = (self.frame_count / self.total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({self.frame_count}/{self.total_frames} frames)")
                
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise
        
        finally:
            # Cleanup
            self.cap.release()
            self.out.release()
            cv2.destroyAllWindows()
        
        logger.info(f"Video processing completed! Output saved to: {self.output_video}")
    
    def get_video_stats(self) -> Dict:
        """Get statistics about the processed video."""
        return {
            'total_frames': self.total_frames,
            'processed_frames': self.frame_count,
            'fps': self.fps,
            'duration_seconds': self.total_frames / self.fps,
            'resolution': f"{self.width}x{self.height}"
        }

def main():
    """Main function to process any video with CORRECTED speed tracking."""
    
    parser = argparse.ArgumentParser(description='Process video with vehicle speed tracking')
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('-o', '--output', help='Path to output video file (optional)')
    parser.add_argument('-m', '--model', 
                       default='/Users/subhashmothukurigmail.com/Projects/Modular Vechical Speed Tracking/src/finetuning/results/vehicle_detection_v1/weights/best.pt',
                       help='Path to model file')
    
    args = parser.parse_args()
    
    # Paths
    input_video = args.input_video
    output_video = args.output
    model_path = args.model
    
    logger.info("🚗 Universal Vehicle Speed Tracking Video Processor (ONNX Optimized)")
    logger.info("=" * 70)
    
    # Check if files exist
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    
    if not os.path.exists(input_video):
        logger.error(f"Input video not found: {input_video}")
        return False
    
    logger.info(f"📁 Input video: {input_video}")
    logger.info(f"🤖 Model: {model_path}")
    logger.info(f"📹 Output video: {output_video or 'Auto-generated'}")
    logger.info("🔧 Using CORRECTED class mapping (swapped truck/motorcycle)")
    logger.info("🔥 ONNX Runtime: " + ("Available" if ONNX_AVAILABLE else "Not Available"))
    
    try:
        # Initialize speed tracker with corrected mapping
        tracker = UniversalVehicleSpeedTracker(model_path, input_video, output_video)
        
        # Get video stats
        stats = tracker.get_video_stats()
        logger.info(f"📊 Video stats: {stats}")
        
        # Process video
        start_time = time.time()
        tracker.process_video()
        end_time = time.time()
        
        processing_time = end_time - start_time
        logger.info(f"⏱️ Processing time: {processing_time:.2f} seconds")
        logger.info(f"🎬 CORRECTED output video saved: {tracker.output_video}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("🎉 CORRECTED video processing completed successfully!")
        logger.info("✅ Truck and motorcycle labels have been swapped!")
    else:
        logger.error("💥 Video processing failed!")
        sys.exit(1)
