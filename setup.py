#!/usr/bin/env python3
"""
Setup script for Vehicle Speed Tracking System
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vehicle-speed-tracking",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A production-ready vehicle detection and speed tracking system using YOLOv8",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vehicle-speed-tracking",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/vehicle-speed-tracking/issues",
        "Documentation": "https://github.com/yourusername/vehicle-speed-tracking/docs",
        "Source Code": "https://github.com/yourusername/vehicle-speed-tracking",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "mkdocs>=1.4.0",
            "mkdocs-material>=9.0.0",
        ],
        "gpu": [
            "torch-audio>=2.0.0",
            "torchaudio>=2.0.0",
        ],
        "optimization": [
            "tensorrt>=8.0.0",
            "openvino>=2023.0.0",
            "coremltools>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vehicle-tracker=finetuning.universal_video_processor:main",
            "live-tracker=finetuning.live_streaming_processor:main",
            "train-model=finetuning.train:main",
            "optimize-model=finetuning.optimization.run_optimizations:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    keywords=[
        "computer vision",
        "object detection",
        "vehicle tracking",
        "speed estimation",
        "YOLOv8",
        "real-time processing",
        "ONNX",
        "machine learning",
        "deep learning",
    ],
)