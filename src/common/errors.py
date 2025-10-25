"""Custom Exception Hierarchy for the Vehicle Speed Tracking Pipeline.

This module defines a hierarchy of custom exceptions used throughout the vehicle
speed tracking application. The hierarchy helps in proper error handling and
recovery at different levels of the application.

Exception Hierarchy:
- AppError (base)
  - ConfigError (configuration issues)
  - VideoIOError (video read/write issues)
  - ModelLoadError (model loading failures)
  - DetectionError (detection failures, recoverable)
  - TrackingError (tracking failures, recoverable)
"""

from __future__ import annotations


class AppError(Exception):
    """Base application error.
    
    This is the root exception class for all custom errors in the application.
    It's useful for catching all application-specific errors at process boundaries
    while letting system errors propagate normally.
    """
    def __init__(self, message: str = "An application error occurred"):
        self.message = message
        super().__init__(self.message)


class ConfigError(AppError):
    """Configuration error indicating invalid or inconsistent settings.
    
    Raised when:
    - Required configuration values are missing
    - Configuration values are invalid
    - Configuration values are inconsistent with each other
    """
    def __init__(self, message: str = "Invalid configuration"):
        super().__init__(f"Configuration error: {message}")


class VideoIOError(AppError):
    """Video input/output operation error.
    
    Raised when:
    - Video file cannot be opened
    - Video capture device is not accessible
    - Frame reading fails
    - Video writer cannot be initialized
    - Video writing fails
    """
    def __init__(self, message: str = "Video I/O operation failed"):
        super().__init__(f"Video I/O error: {message}")


class ModelLoadError(AppError):
    """Model loading or initialization error.
    
    Raised when:
    - Model weights file is missing
    - Model weights are corrupted
    - Model is incompatible with runtime environment
    - Model initialization fails
    """
    def __init__(self, message: str = "Failed to load model"):
        super().__init__(f"Model load error: {message}")


class DetectionError(AppError):
    """Object detection inference error (recoverable).
    
    Raised when:
    - Detection inference fails
    - Detection output is invalid
    - GPU/CPU errors during inference
    
    This is considered recoverable - the application can skip the current
    frame and continue processing.
    """
    def __init__(self, message: str = "Detection inference failed"):
        super().__init__(f"Detection error: {message}")


class TrackingError(AppError):
    """Object tracking update error (recoverable).
    
    Raised when:
    - Tracker state update fails
    - Invalid detection input to tracker
    - Tracker internal state corruption
    
    This is considered recoverable - the application can skip the current
    frame and continue processing.
    """
    def __init__(self, message: str = "Tracking update failed"):
        super().__init__(f"Tracking error: {message}")

