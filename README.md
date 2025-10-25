# 🚗 Vehicle Speed Tracking System

A production-ready vehicle detection and speed tracking system using YOLOv8, optimized for both offline video analysis and live streaming applications.

## 🌟 Features

- **High Accuracy**: 87.2% mAP50 with fine-tuned YOLOv8 model
- **Real-time Processing**: Optimized for live video streams
- **Speed Tracking**: Accurate vehicle speed calculation using ByteTracker
- **Multiple Vehicle Types**: Detection of cars, trucks, and motorcycles
- **ONNX Optimization**: Export to ONNX for production deployment
- **GPU Acceleration**: Support for CUDA and MPS (Apple Silicon)
- **Live Streaming**: Real-time camera feed processing
- **Batch Processing**: Efficient offline video analysis

## 📊 Performance

| Model | mAP50 | Precision | Recall | FPS | Use Case |
|-------|-------|-----------|--------|-----|----------|
| **Fine-tuned YOLOv8n** | 87.2% | 89.8% | 85.0% | 13.45 | Offline processing |
| **Live Streaming** | 87.2% | 89.8% | 85.0% | 5.52 | Batch processing |
| **Ultra Optimized** | 87.2% | 89.8% | 85.0% | 4.60 | Fast batch processing |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vehicle-speed-tracking.git
cd vehicle-speed-tracking

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### Offline Video Processing
```bash
python src/finetuning/universal_video_processor.py data/input_video.mp4 -o data/output_video.mp4
```

#### Live Streaming (Webcam)
```bash
python src/finetuning/live_streaming_processor.py --source 0 --target-fps 30
```

#### Live Streaming (Video File)
```bash
python src/finetuning/live_streaming_processor.py --source data/input_video.mp4 --output data/output_video.mp4
```

## 📁 Project Structure

```
vehicle-speed-tracking/
├── README.md                           # This file
├── LICENSE                             # MIT License
├── requirements.txt                    # Python dependencies
├── setup.py                           # Package setup
├── data/                              # Data directory
│   ├── data.yaml                      # YOLOv8 dataset configuration
│   ├── images/                        # Training images
│   │   ├── train/                     # Training set
│   │   ├── val/                       # Validation set
│   │   └── test/                      # Test set
│   └── labels/                        # YOLO format labels
│       ├── train/                     # Training labels
│       ├── val/                       # Validation labels
│       └── test/                      # Test labels
├── src/                               # Source code
│   ├── finetuning/                    # Fine-tuning module
│   │   ├── train.py                   # Training script
│   │   ├── universal_video_processor.py  # Universal video processor
│   │   ├── live_streaming_processor.py   # Live streaming processor
│   │   ├── evaluate_model.py          # Model evaluation
│   │   ├── optimization/              # Optimization tools
│   │   │   ├── accuracy_optimizer.py  # Accuracy optimization
│   │   │   ├── realtime_optimizer.py  # Real-time optimization
│   │   │   └── run_optimizations.py   # Optimization runner
│   │   └── results/                   # Training results
│   │       └── vehicle_detection_v1/  # Model weights and metrics
│   ├── detectors/                     # Detection modules
│   ├── tracking/                      # Tracking algorithms
│   ├── pipeline/                      # Processing pipeline
│   └── utils/                         # Utility functions
├── docs/                              # Documentation
│   ├── api/                           # API documentation
│   ├── tutorials/                     # Tutorials and guides
│   └── production_guide.md            # Production deployment guide
└── tests/                             # Test files
```

## 🛠️ Development

### Training Your Own Model

1. **Prepare Data**: Organize your images and labels in YOLO format
2. **Configure Dataset**: Update `data/data.yaml` with your class names
3. **Start Training**: Run the training script

```bash
cd src/finetuning
python train.py
```

### Model Optimization

#### Accuracy Optimization
```bash
python optimization/accuracy_optimizer.py
```

#### Real-time Optimization
```bash
python optimization/realtime_optimizer.py
```

#### Run All Optimizations
```bash
python optimization/run_optimizations.py --all
```

## 📈 Model Performance

### Training Results
- **Epochs**: 100
- **Best mAP50**: 87.51% (Epoch 93)
- **Best mAP50-95**: 72.47% (Epoch 82)
- **Best Precision**: 93.85% (Epoch 2)
- **Best Recall**: 85.33% (Epoch 79)

### Class Distribution
- **Car**: Most common vehicle type
- **Truck**: Commercial vehicles
- **Motorcycle**: Two-wheeled vehicles

## 🎯 Production Deployment

### Current Status
- ✅ **Offline Processing**: Production ready
- ✅ **Batch Analysis**: Production ready
- ⚠️ **Live Streaming**: Needs optimization for 30+ FPS

### Optimization Strategies
1. **Model Quantization**: INT8 quantization for 2-3x speedup
2. **TensorRT Integration**: NVIDIA GPU optimization
3. **ONNX Runtime**: Optimized inference engine
4. **Hardware Upgrade**: High-end GPU/CPU

### Deployment Options
- **Docker Container**: Containerized deployment
- **Cloud Services**: AWS, GCP, Azure
- **Edge Devices**: NVIDIA Jetson, Intel NUC
- **Local Server**: High-performance workstation

## 🔧 Configuration

### Model Configuration
```yaml
# data/data.yaml
path: /path/to/your/data
train: images/train
val: images/val
test: images/test

names:
  0: car
  1: truck
  2: motorcycle
```

### Live Streaming Configuration
```bash
# High performance settings
--target-fps 30 --frame-skip 1 --input-size 640 --use-gpu

# Balanced settings
--target-fps 15 --frame-skip 2 --input-size 320

# Ultra-fast settings
--target-fps 10 --frame-skip 5 --input-size 224
```

## 📚 API Reference

### UniversalVideoProcessor
```python
from src.finetuning.universal_video_processor import UniversalVehicleSpeedTracker

processor = UniversalVehicleSpeedTracker(
    model_path="path/to/model.pt",
    input_video="input.mp4",
    output_video="output.mp4"
)
processor.process_video()
```

### LiveStreamingProcessor
```python
from src.finetuning.live_streaming_processor import LiveStreamingProcessor

processor = LiveStreamingProcessor(
    model_path="path/to/model.pt",
    target_fps=30,
    frame_skip=1,
    input_size=640
)
processor.process_live_stream(source=0)
```

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/

# Test model performance
python src/finetuning/evaluate_model.py

# Test live streaming
python src/finetuning/live_streaming_processor.py --source 0
```

## 📊 Benchmarks

### Hardware Requirements
- **Minimum**: CPU with 8GB RAM
- **Recommended**: GPU with 4GB VRAM
- **Optimal**: RTX 3080+ or equivalent

### Performance Metrics
- **Inference Time**: ~45ms per frame (640x640)
- **Memory Usage**: ~2GB RAM
- **Model Size**: 5.9MB (PyTorch), 12.3MB (ONNX)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the base model
- [Supervision](https://github.com/roboflow/supervision) for tracking utilities
- [OpenCV](https://opencv.org/) for video processing
- [ONNX Runtime](https://onnxruntime.ai/) for model optimization

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/vehicle-speed-tracking/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/vehicle-speed-tracking/discussions)
- **Email**: your.email@example.com

## 🔗 Links

- **Documentation**: [Full Documentation](docs/)
- **Production Guide**: [Production Deployment](docs/production_guide.md)
- **API Reference**: [API Documentation](docs/api/)
- **Tutorials**: [Getting Started](docs/tutorials/)

---

**Made with ❤️ for the computer vision community**
# Traffic_Analysis
