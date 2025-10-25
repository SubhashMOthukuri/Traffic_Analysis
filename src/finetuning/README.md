# YOLOv8 Fine-tuning Module

This module provides a clean and organized way to fine-tune YOLOv8 models for vehicle detection in your speed tracking project.

## 🚀 Quick Start

### Option 1: One-Click Setup
```bash
cd "/Users/subhashmothukurigmail.com/Projects/Modular Vechical Speed Tracking"
./src/finetuning/setup_and_train.sh
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip3 install -r src/finetuning/requirements.txt

# Start training
python3 src/finetuning/train.py
```

## 📁 Directory Structure

```
src/finetuning/
├── __init__.py          # Main fine-tuning class
├── train.py             # Training script
├── requirements.txt     # Dependencies
├── config.yaml         # Configuration file
├── setup_and_train.sh  # One-click setup script
├── models/             # Saved models (created during training)
└── results/            # Training results (created during training)
```

## 🎯 Features

- **Transfer Learning**: Uses pretrained YOLOv8 weights
- **Automatic Environment Setup**: Detects CPU/GPU automatically
- **Data Augmentation**: Built-in augmentation for better performance
- **Early Stopping**: Prevents overfitting
- **Model Validation**: Automatic validation after training
- **Clean Logging**: Comprehensive logging and progress tracking

## 📊 Dataset Information

- **Classes**: car, truck, motorcycle
- **Train Images**: 160
- **Validation Images**: 30
- **Test Images**: 11
- **Total Annotations**: 82,335

## ⚙️ Configuration

Edit `config.yaml` to customize:
- Model size (yolov8n.pt to yolov8x.pt)
- Training parameters
- Data augmentation settings
- Output configuration

## 🔧 Available Models

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| yolov8n.pt | 6MB | Fastest | Good | Real-time detection |
| yolov8s.pt | 22MB | Fast | Better | Balanced performance |
| yolov8m.pt | 50MB | Medium | High | High accuracy needed |
| yolov8l.pt | 87MB | Slow | Very High | Best accuracy |
| yolov8x.pt | 136MB | Slowest | Highest | Maximum accuracy |

## 📈 Expected Results

- **Training Time**: 2-4 hours (CPU) / 30-60 minutes (GPU)
- **Model Performance**: Good detection for cars/trucks, moderate for motorcycles
- **Output**: Best model saved in `results/vehicle_detection/weights/best.pt`

## 🎉 After Training

Your trained model will be saved and ready to use in your vehicle speed tracking application. The model can be loaded using:

```python
from ultralytics import YOLO
model = YOLO('src/finetuning/results/vehicle_detection/weights/best.pt')
```
