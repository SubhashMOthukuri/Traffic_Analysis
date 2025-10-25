"""Video input/output utility functions."""

from __future__ import annotations
import cv2
import logging
from typing import Optional, Tuple
from src.common.errors import VideoIOError


logger = logging.getLogger(__name__)


def open_video(source: str) -> cv2.VideoCapture:
    """Open a video source and return VideoCapture object."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise VideoIOError(f"Could not open video source: {source}")
    return cap


def get_frame_size_fps(cap: cv2.VideoCapture) -> Tuple[int, int, float]:
    """Get frame width, height and FPS from video capture."""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        logger.warning("FPS unknown; defaulting to 30.0 for speed calculations.")
        fps = 30.0
    return width, height, fps


def make_video_writer(path: str, fps: float, size: Tuple[int, int]) -> Optional[cv2.VideoWriter]:
    """Create a VideoWriter for saving output video."""
    if not path:
        return None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if not writer.isOpened():
        raise VideoIOError(f"Could not open VideoWriter at: {path}")
    return writer