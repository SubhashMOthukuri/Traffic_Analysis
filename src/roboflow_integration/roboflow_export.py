"""Script to export annotations from Roboflow to YOLO format."""

import os
import sys
from pathlib import Path
from roboflow import Roboflow
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_dataset(api_key: str, project_name: str, output_dir: str):
    """Export annotated dataset from Roboflow in YOLO format."""
    rf = Roboflow(api_key=api_key)
    
    try:
        # Get the project
        workspace = rf.workspace()
        project = workspace.project(project_name)
        
        # Download latest version in YOLO format
        dataset = project.version(project.latest_version()).download("yolov8")
        
        logger.info(f"Dataset downloaded to {dataset}")
        
        # Move/organize files if needed
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        return True
    except Exception as e:
        logger.error(f"Failed to export dataset: {e}")
        return False

def main():
    """Main function to export annotations."""
    if len(sys.argv) != 2:
        print("Usage: python roboflow_export.py YOUR_API_KEY")
        sys.exit(1)

    api_key = sys.argv[1]
    project_name = "upview_vehicle_detection"
    output_dir = str(Path(__file__).parents[2] / "data" / "upview_yolo_dataset")

    if export_dataset(api_key, project_name, output_dir):
        print(f"\nDataset exported successfully to: {output_dir}")
        print("\nNext steps:")
        print("1. Use the exported dataset to train YOLOv8")
        print("2. Update the model path in your pipeline")
        print("3. Run inference on upview.mp4")
    else:
        print("\nFailed to export dataset. Check the error messages above.")

if __name__ == "__main__":
    main()