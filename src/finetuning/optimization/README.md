# 🚀 Model Optimization Toolkit

This folder contains comprehensive optimization tools for improving YOLOv8 model performance in both accuracy and real-time inference.

## 📁 Files Overview

### 🎯 Accuracy Optimization
- **`accuracy_optimizer.py`** - Boosts mAP/precision/recall through data improvements and model tweaks

### ⚡ Real-Time Optimization  
- **`realtime_optimizer.py`** - Optimizes for 30-60 FPS live video feeds with speed/latency improvements

### 🔧 Main Runner
- **`run_optimizations.py`** - Comprehensive optimization runner with command-line interface

## 🎯 Accuracy Optimization Features

### 🔹 Data Improvements
- **Strong Augmentations**: Enable mosaic, mixup, HSV, rotation, scaling, perspective
- **Class Balance Analysis**: Analyze and visualize class distribution
- **Higher Resolution**: Train with 800x800 for better small object detection

### 🔹 Model/Training Tweaks
- **Longer Fine-tuning**: Additional epochs with lower learning rate
- **Larger Models**: YOLOv8s/YOLOv8m for +3-5 mAP improvement
- **Advanced Augmentations**: Copy-paste, random erasing, geometric transforms

## ⚡ Real-Time Optimization Features

### 🔹 Model Compression
- **ONNX Export**: Dynamic shapes + half precision for 2x speed boost
- **Quantization**: INT8/FP16 quantization for smaller models
- **Pruning**: Remove low-weight channels to shrink model size

### 🔹 Hardware/Runtime
- **Mac MPS Support**: Metal Performance Shaders acceleration
- **ONNX Runtime**: ~2x speed boost over raw PyTorch
- **Batch Size 1**: Optimized for streaming inference

### 🔹 Pipeline Tricks
- **Frame Skipping**: Process every Nth frame for 60fps streams
- **Async Processing**: Threaded/asyncio inference
- **Smart Resizing**: Resize frames before inference

## 🚀 Quick Start

### Run All Optimizations
```bash
cd src/finetuning/optimization
python3 run_optimizations.py --all
```

### Run Specific Optimizations
```bash
# Accuracy optimization only
python3 run_optimizations.py --accuracy

# Real-time optimization only  
python3 run_optimizations.py --realtime
```

### Individual Scripts
```bash
# Accuracy optimization
python3 accuracy_optimizer.py

# Real-time optimization
python3 realtime_optimizer.py
```

## 📊 Expected Improvements

### 🎯 Accuracy Optimizations
- **mAP50**: +3-8% improvement
- **Small Objects**: Better detection with higher resolution
- **Generalization**: Stronger with advanced augmentations
- **Class Balance**: Improved with analysis and balancing

### ⚡ Real-Time Optimizations
- **Inference Speed**: 2-4x faster with ONNX + MPS
- **Memory Usage**: 30-50% reduction with quantization
- **FPS**: 30-60 FPS achievable on Mac M3 Pro
- **Latency**: <33ms per frame for real-time processing

## 🔧 Configuration

### Accuracy Optimization Config
```python
augmentation_config = {
    'degrees': 15.0,           # Rotation degrees
    'translate': 0.2,          # Translation fraction
    'scale': 0.9,              # Scale range
    'hsv_h': 0.015,           # HSV-Hue augmentation
    'hsv_s': 0.7,             # HSV-Saturation augmentation
    'mosaic': 1.0,            # Mosaic probability
    'mixup': 0.15,            # Mixup probability
    'copy_paste': 0.3,        # Copy-paste probability
}
```

### Real-Time Optimization Config
```python
realtime_config = {
    'skip_frames': 1,          # Process every 2nd frame
    'resize_factor': 1.0,      # Resize input (0.5 = half size)
    'device': 'mps',          # Use MPS on Mac
    'batch_size': 1,          # Single frame processing
    'half_precision': True,    # FP16 for speed
}
```

## 📈 Performance Monitoring

### Benchmarking
- **Inference Time**: Average, min, max, std deviation
- **FPS**: Frames per second calculation
- **Memory Usage**: RAM consumption tracking
- **GPU Utilization**: MPS/GPU usage monitoring

### Reports Generated
- `accuracy_optimization_report.json` - Accuracy improvements
- `realtime_optimization_report.json` - Speed/latency metrics
- `production_config.yaml` - Production deployment config

## 🏭 Production Deployment

### Optimized Model Usage
```python
# Load optimized ONNX model
import onnxruntime as ort
session = ort.InferenceSession('model_optimized.onnx')

# Real-time processing
def process_frame(frame):
    # Resize for speed
    frame = cv2.resize(frame, (640, 640))
    
    # Run inference
    results = session.run(None, {'images': frame})
    
    return results
```

### Mac MPS Acceleration
```python
# Use MPS for acceleration
model = YOLO('model.pt')
results = model(frame, device='mps')
```

## 🎯 Best Practices

### For Accuracy
1. **Start with strong augmentations** - Biggest impact on generalization
2. **Analyze class balance** - Address imbalanced datasets
3. **Try higher resolution** - Better for small objects
4. **Fine-tune longer** - Additional epochs with lower LR

### For Real-Time
1. **Export to ONNX** - Essential for production speed
2. **Use MPS on Mac** - Leverage Metal Performance Shaders
3. **Skip frames** - Process every 2nd frame for 60fps streams
4. **Batch size 1** - Optimize for single frame processing

## 🔍 Troubleshooting

### Common Issues
- **MPS not available**: Use CPU fallback
- **ONNX export fails**: Check model compatibility
- **Low FPS**: Reduce input resolution or skip frames
- **Memory issues**: Use quantization or smaller batch size

### Performance Tips
- **Warm up model**: Run 10-20 inference cycles before benchmarking
- **Use async processing**: For multiple video streams
- **Monitor memory**: Watch for memory leaks in long-running processes
- **Profile bottlenecks**: Use profiling tools to identify slow operations

## 📚 References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Mac MPS](https://developer.apple.com/metal/Metal-Performance-Shaders/)
- [PyTorch Optimization](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

**🎉 Ready to optimize your vehicle detection model for maximum performance!**
