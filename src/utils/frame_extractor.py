"""Utility to extract frames from video at specified intervals."""

import cv2
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_frames(video_path: str, output_dir: str, fps: float = 1.0):
    """
    Extract frames from video at specified fps rate.
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        fps: Number of frames to extract per second
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return False
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video FPS: {video_fps}")
    logger.info(f"Extracting 1 frame every {frame_interval} frames")
    logger.info(f"Total frames in video: {total_frames}")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame at specified interval
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_count+1:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            if saved_count % 10 == 0:
                logger.info(f"Saved {saved_count} frames")
                
        frame_count += 1
    
    cap.release()
    logger.info(f"Extracted {saved_count} frames at {fps} fps")
    return True

def main():
    """Extract frames from the upview video."""
    root_dir = Path(__file__).parents[2]
    video_path = str(root_dir / "data" / "upview.mp4")
    output_dir = str(root_dir / "data" / "upview_frames")
    
    # Extract 1 frame per second
    extract_frames(video_path, output_dir, fps=1.0)

if __name__ == "__main__":
    main()