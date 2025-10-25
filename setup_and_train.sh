#!/bin/bash
# YOLOv8 Training Setup Script

echo "🚀 Setting up YOLOv8 Training Environment..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python3 found: $(python3 --version)"

# Install dependencies
echo "📦 Installing training dependencies..."
pip3 install -r training_requirements.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Make training script executable
chmod +x train_yolov8.py

echo "🎯 Starting YOLOv8 training..."
echo "📊 Dataset Summary:"
echo "   - Total Images: 201"
echo "   - Train Split: 160 images"
echo "   - Validation Split: 30 images"
echo "   - Test Split: 11 images"
echo "   - Classes: car, truck, bus"
echo ""

# Start training
python3 train_yolov8.py

echo "🏁 Training process completed!"
