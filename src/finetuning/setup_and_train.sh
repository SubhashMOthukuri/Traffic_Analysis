#!/bin/bash
# YOLOv8 Fine-tuning Setup and Training Script

echo "🚀 YOLOv8 Fine-tuning Setup for Vehicle Detection"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python3 found: $(python3 --version)"

# Navigate to project directory
PROJECT_DIR="/Users/subhashmothukurigmail.com/Projects/Modular Vechical Speed Tracking"
cd "$PROJECT_DIR"

# Install dependencies
echo "📦 Installing fine-tuning dependencies..."
pip3 install -r src/finetuning/requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Make training script executable
chmod +x src/finetuning/train.py

echo ""
echo "📊 Dataset Summary:"
echo "  Classes: car, truck, motorcycle"
echo "  Train images: 160"
echo "  Validation images: 30"
echo "  Test images: 11"
echo "  Total annotations: 82,335"
echo ""

echo "🎯 Starting YOLOv8 fine-tuning..."
echo "This will use transfer learning with pretrained weights."
echo "Training may take 2-4 hours on CPU or 30-60 minutes on GPU."
echo ""

# Start training
python3 src/finetuning/train.py

echo ""
echo "🏁 Fine-tuning process completed!"
echo "Check the results in: src/finetuning/results/"
