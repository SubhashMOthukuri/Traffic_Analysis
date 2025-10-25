# API Documentation

## Core Classes

### UniversalVehicleSpeedTracker

The main class for offline video processing with vehicle detection and speed tracking.

#### Constructor
```python
UniversalVehicleSpeedTracker(model_path: str, input_video: str, output_video: str = None)
```

**Parameters:**
- `model_path` (str): Path to the YOLOv8 model weights (.pt file)
- `input_video` (str): Path to the input video file
- `output_video` (str, optional): Path for the output video. If None, auto-generated

#### Methods

##### `process_video()`
Process the entire video and generate output with speed tracking.

**Returns:** None

**Example:**
```python
processor = UniversalVehicleSpeedTracker("model.pt", "input.mp4", "output.mp4")
processor.process_video()
```

##### `get_video_stats() -> Dict`
Get statistics about the processed video.

**Returns:** Dictionary containing video statistics

**Example:**
```python
stats = processor.get_video_stats()
print(f"Total frames: {stats['total_frames']}")
print(f"FPS: {stats['fps']}")
```

### LiveStreamingProcessor

Production-ready processor for live video streaming with real-time optimizations.

#### Constructor
```python
LiveStreamingProcessor(
    model_path: str,
    target_fps: int = 30,
    frame_skip: int = 1,
    input_size: int = 320,
    use_gpu: bool = True,
    async_processing: bool = True
)
```

**Parameters:**
- `model_path` (str): Path to the YOLOv8 model weights
- `target_fps` (int): Target FPS for processing (default: 30)
- `frame_skip` (int): Process every Nth frame (default: 1)
- `input_size` (int): Input image size for inference (default: 320)
- `use_gpu` (bool): Enable GPU acceleration (default: True)
- `async_processing` (bool): Enable async processing (default: True)

#### Methods

##### `process_live_stream(source: int = 0, output_path: str = None)`
Process live video stream with real-time optimizations.

**Parameters:**
- `source` (int): Video source (0 for webcam, file path for video)
- `output_path` (str, optional): Path to save output video

**Example:**
```python
processor = LiveStreamingProcessor("model.pt", target_fps=30)
processor.process_live_stream(source=0)  # Webcam
```

##### `process_video_file(input_path: str, output_path: str = None)`
Process video file with live streaming optimizations.

**Parameters:**
- `input_path` (str): Path to input video file
- `output_path` (str, optional): Path for output video

**Example:**
```python
processor.process_video_file("input.mp4", "output.mp4")
```

## Utility Functions

### Model Training

#### `train_yolov8_model()`
Train a YOLOv8 model on custom vehicle data.

**Parameters:**
- `data_yaml` (str): Path to dataset configuration file
- `epochs` (int): Number of training epochs
- `batch_size` (int): Training batch size
- `imgsz` (int): Input image size

**Example:**
```python
from finetuning.train import train_yolov8_model

results = train_yolov8_model(
    data_yaml="data/data.yaml",
    epochs=100,
    batch_size=16,
    imgsz=640
)
```

### Model Optimization

#### `AccuracyOptimizer`
Class for accuracy optimization techniques.

**Methods:**
- `fine_tune_longer()`: Continue training with lower learning rate
- `try_larger_input_size()`: Train with larger input resolution
- `try_larger_yolov8_model()`: Use larger YOLOv8 variants

#### `RealtimeOptimizer`
Class for real-time performance optimization.

**Methods:**
- `export_to_onnx()`: Export model to ONNX format
- `optimize_for_mps()`: Configure for Apple Metal acceleration
- `measure_inference_speed()`: Benchmark inference performance

## Configuration

### Dataset Configuration (data.yaml)
```yaml
path: /path/to/your/data
train: images/train
val: images/val
test: images/test

names:
  0: car
  1: truck
  2: motorcycle
```

### Production Configuration
```yaml
model:
  path: path/to/model.pt
  device: mps
  batch_size: 1
  imgsz: 640

optimization:
  skip_frames: 1
  resize_factor: 1.0
  async_processing: true

performance:
  target_fps: 30
  max_latency_ms: 33
```

## Error Handling

### Common Exceptions

#### `ValueError`
Raised when video file cannot be opened or model path is invalid.

#### `RuntimeError`
Raised when ONNX session is not properly initialized.

#### `IOError`
Raised when video writer cannot be created.

### Error Handling Example
```python
try:
    processor = UniversalVehicleSpeedTracker("model.pt", "input.mp4")
    processor.process_video()
except ValueError as e:
    print(f"Configuration error: {e}")
except RuntimeError as e:
    print(f"Runtime error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Tips

### For Maximum Accuracy
- Use larger input size (640x640 or higher)
- Process every frame (frame_skip=1)
- Use GPU acceleration
- Fine-tune on domain-specific data

### For Maximum Speed
- Use smaller input size (224x224 or 320x320)
- Skip frames (frame_skip=2 or higher)
- Enable ONNX optimization
- Use async processing

### For Balanced Performance
- Use medium input size (320x320)
- Skip every other frame (frame_skip=2)
- Enable GPU acceleration
- Use optimized model weights
