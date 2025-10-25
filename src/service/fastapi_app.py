"""FastAPI service for vehicle speed tracking."""

from __future__ import annotations
import asyncio
import logging
from typing import Optional
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import (
    VisualConfig,
    SpeedConfig,
    RuntimeConfig,
    DetectorConfig,
    TrackerConfig,
)
from src.detectors.yolo_detector import YOLODetector
from src.tracking.byte_tracker import ByteTrackerWrapper
from src.pipeline.speed_tracker import SpeedTracker
from src.common.errors import AppError

# Configure logging
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Vehicle Speed Tracking API",
    description="API for detecting and tracking vehicle speeds in video frames",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Concurrency control
MAX_CONCURRENCY = 4
sem = asyncio.Semaphore(MAX_CONCURRENCY)

# Global pipeline components
_detector: Optional[YOLODetector] = None
_tracker: Optional[ByteTrackerWrapper] = None
_pipe: Optional[SpeedTracker] = None


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline components on service startup."""
    global _detector, _tracker, _pipe

    try:
        # Load configurations
        det_cfg = DetectorConfig()
        trk_cfg = TrackerConfig()
        visual = VisualConfig()
        speed_cfg = SpeedConfig()
        runtime = RuntimeConfig()

        # Initialize components
        _detector = YOLODetector(
            model_path=det_cfg.model_path,
            conf_threshold=det_cfg.conf_threshold,
            target_classes=list(det_cfg.target_classes),
            warmup=True
        )
        _tracker = ByteTrackerWrapper(
            frame_rate=int(runtime.frame_rate),
            track_activation_threshold=trk_cfg.track_activation_threshold,
            lost_track_seconds=trk_cfg.lost_track_seconds,
            minimum_matching_threshold=trk_cfg.minimum_matching_threshold,
        )
        _pipe = SpeedTracker(visual, speed_cfg, runtime)
        
        logger.info("Service initialized successfully: model=%s", det_cfg.model_path)
    except Exception as e:
        logger.error("Service initialization failed: %s", e, exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on service shutdown."""
    global _detector, _tracker, _pipe
    _detector = None
    _tracker = None
    _pipe = None
    logger.info("Service shutdown complete")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "components": {
            "detector": _detector is not None,
            "tracker": _tracker is not None,
            "pipeline": _pipe is not None,
        }
    }


@app.get("/metrics")
async def metrics():
    """Basic metrics endpoint."""
    return {
        "max_concurrency": MAX_CONCURRENCY,
        "current_requests": MAX_CONCURRENCY - sem._value
    }


@app.post("/infer")
async def infer(image: UploadFile = File(...)):
    """Process an image frame and return speed tracking results.
    
    Args:
        image: Uploaded image file (JPEG/PNG)
        
    Returns:
        dict: Processed image bytes (hex) and dimensions
        
    Raises:
        HTTPException: For various error conditions
    """
    if _detector is None or _tracker is None or _pipe is None:
        raise HTTPException(
            status_code=503,
            detail="Service components not initialized"
        )

    # Backpressure handling
    if not sem.locked() and sem._value == 0:
        raise HTTPException(
            status_code=429,
            detail="Server is at maximum capacity"
        )

    async with sem:
        try:
            # Read and decode image
            data = await image.read()
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid image format or corrupted data"
                )

            # Process frame
            dets = _detector.detect(frame)
            tracked = _tracker.update(dets)
            annotated = _pipe.step(frame, tracked)

            # Encode result
            ok, jpg = cv2.imencode('.jpg', annotated)
            if not ok:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to encode output image"
                )

            return {
                "bytes": jpg.tobytes().hex(),
                "width": int(annotated.shape[1]),
                "height": int(annotated.shape[0])
            }

        except HTTPException:
            raise
        except AppError as e:
            logger.error("Application error in /infer: %s", e)
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.exception("Unexpected error in /infer: %s", e)
            raise HTTPException(
                status_code=500,
                detail="Internal server error"
            )


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    uvicorn.run(
        "src.service.fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )