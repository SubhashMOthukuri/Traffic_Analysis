#!/usr/bin/env python3
"""
Test script to check class detection and mapping
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_class_detection():
    """Test what classes the model is detecting."""
    
    project_root = "/Users/subhashmothukurigmail.com/Projects/Modular Vechical Speed Tracking"
    model_path = f"{project_root}/src/finetuning/results/vehicle_detection_v1/weights/best.pt"
    input_video = f"{project_root}/data/upview.mp4"
    
    # Load model
    model = YOLO(model_path)
    logger.info(f"Model class names: {model.names}")
    
    # Open video and get first frame
    cap = cv2.VideoCapture(input_video)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        logger.error("Could not read video frame")
        return
    
    # Run detection
    results = model(frame, conf=0.3, verbose=False)[0]
    
    logger.info("Detection results:")
    for i, r in enumerate(results.boxes):
        x1, y1, x2, y2 = r.xyxy[0].cpu().numpy().astype(int)
        conf = r.conf[0].cpu().numpy()
        cls = int(r.cls[0].cpu().numpy())
        
        class_name = model.names[cls]
        logger.info(f"Detection {i+1}: class_id={cls}, class_name='{class_name}', conf={conf:.3f}")
        logger.info(f"  Bbox: ({x1}, {y1}, {x2}, {y2})")
    
    # Check if there's a class mapping issue
    logger.info("\nChecking for potential class mapping issues...")
    
    # Count detections by class
    class_counts = {}
    for r in results.boxes:
        cls = int(r.cls[0].cpu().numpy())
        class_name = model.names[cls]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    logger.info(f"Detection counts by class: {class_counts}")

if __name__ == "__main__":
    test_class_detection()
