"""Roboflow integration for annotating upview frames and exporting YOLO format."""

import os
import sys
import time
from pathlib import Path
from roboflow import Roboflow
import logging
import shutil
import json

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RoboflowAnnotator:
    def __init__(self, api_key: str):
        """Initialize Roboflow with API key."""
        try:
            logger.info("Initializing Roboflow connection...")
            self.rf = Roboflow(api_key=api_key)
            # Test the connection
            self.rf.workspace()
            logger.info("Successfully connected to Roboflow")
        except Exception as e:
            logger.error(f"Failed to connect to Roboflow: {str(e)}")
            raise
        self.project = None
        self.workspace = None

    def create_project(self, name: str):
        """Create a new Roboflow project."""
        try:
            logger.info(f"Creating project '{name}'...")
            self.workspace = self.rf.workspace()
            self.project = self.workspace.create_project(
                project_name=name,
                project_type="object-detection",
                project_license="CC BY 4.0",
                annotation="object-detection"
            )
            logger.info(f"Created project: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create project: {str(e)}")
            if "already exists" in str(e).lower():
                logger.info("Attempting to use existing project...")
                try:
                    self.project = self.workspace.project(name)
                    logger.info("Successfully connected to existing project")
                    return True
                except Exception as e2:
                    logger.error(f"Failed to connect to existing project: {str(e2)}")
            return False

    def upload_images(self, image_dir: str, batch_size: int = 50):
        """Upload images to Roboflow project in batches."""
        if not self.project:
            logger.error("No project initialized. Call create_project first.")
            return False

        try:
            image_paths = list(Path(image_dir).glob("*.jpg"))
            total_images = len(image_paths)
            if total_images == 0:
                logger.error(f"No jpg images found in {image_dir}")
                return False
            
            logger.info(f"Found {total_images} images to upload from {image_dir}")

            for i in range(0, total_images, batch_size):
                batch = image_paths[i:i + batch_size]
                batch_num = i//batch_size + 1
                logger.info(f"Uploading batch {batch_num} of {(total_images-1)//batch_size + 1} ({len(batch)} images)")
                for img_path in batch:
                    try:
                        self.project.upload(str(img_path))
                        logger.debug(f"Uploaded {img_path.name}")
                    except Exception as e:
                        logger.error(f"Failed to upload {img_path.name}: {str(e)}")
                        continue
                time.sleep(1)  # Rate limiting
                logger.info(f"Completed batch {batch_num}")

            return True
        except Exception as e:
            logger.error(f"Failed to upload images: {str(e)}")
            return False

    def generate_yolo_dataset(self, output_dir: str):
        """Export annotations in YOLO format."""
        if not self.project:
            logger.error("No project initialized")
            return False

        try:
            # Export the dataset in YOLO format
            dataset = self.project.version(1).download("yolov8")
            
            # Create directory structure
            yolo_dir = Path(output_dir)
            train_dir = yolo_dir / "images" / "train"
            train_dir.mkdir(parents=True, exist_ok=True)
            
            # Move files to correct structure
            for f in Path(dataset).glob("**/train/images/*"):
                shutil.copy2(f, train_dir)
            
            logger.info(f"Dataset exported to {output_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to generate YOLO dataset: {e}")
            return False

def main():
    """Main function to run the annotation workflow."""
    if len(sys.argv) != 2:
        logger.error("Usage: python roboflow_annotate.py YOUR_API_KEY")
        sys.exit(1)

    api_key = sys.argv[1]
    frames_dir = str(Path(__file__).parents[2] / "data" / "upview_frames")
    output_dir = str(Path(__file__).parents[2] / "data" / "upview_yolo_dataset")

    try:
        logger.info("Starting Roboflow annotation workflow...")
        annotator = RoboflowAnnotator(api_key)
    except Exception as e:
        logger.error("Failed to initialize Roboflow. Check your API key and internet connection.")
        sys.exit(1)
    
    # Create project
    if not annotator.create_project("upview_vehicle_detection"):
        logger.error("Failed to create or access project")
        sys.exit(1)
    
    # Upload images
    if not annotator.upload_images(frames_dir):
        logger.error("Failed to upload images")
        sys.exit(1)
    
    logger.info("\nImages uploaded successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Go to https://app.roboflow.com")
    logger.info("2. Find your project 'upview_vehicle_detection'")
    logger.info("3. Label your images using Roboflow's annotation tool")
    logger.info("4. Generate a new version of your dataset")
    logger.info("5. Run roboflow_export.py to download annotations in YOLO format")
    logger.info(f"\nAfter annotating, the dataset will be exported to: {output_dir}")

if __name__ == "__main__":
    main()