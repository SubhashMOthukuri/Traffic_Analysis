"""YOLO object detector wrapper with filtering capabilities."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import logging
from ultralytics import YOLO
import supervision as sv
import numpy as np
from ..common.errors import ModelLoadError, DetectionError


logger = logging.getLogger(__name__)


@dataclass
class YOLODetector:
    """Wrapper for YOLO detector with class filtering and error handling."""
    model_path: str
    conf_threshold: float
    target_classes: List[str]
    warmup: bool = True

    def __post_init__(self) -> None:
        """Initialize YOLO model and optionally run warmup inference."""
        try:
            logger.info("Loading YOLO model: %s", self.model_path)
            self.model = YOLO(self.model_path)
            self._class_map = self.model.names

            if self.warmup:
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = self.model(dummy, conf=self.conf_threshold, verbose=False, imgsz=640)
        except Exception as e:
            raise ModelLoadError(f"Failed to load YOLO model '{self.model_path}': {e}") from e

    def detect(self, frame, imgsz: int = 640) -> sv.Detections:
        """Run inference on frame and filter for target classes."""
        try:
            results = self.model(frame, conf=self.conf_threshold, verbose=False, imgsz=imgsz)[0]
        except Exception as e:
            raise DetectionError(f"YOLO inference failed: {e}") from e

        dets = sv.Detections.from_ultralytics(results)
        if dets.class_id is None or len(dets) == 0:
            return sv.Detections.empty()

        keep = []
        for i, cid in enumerate(dets.class_id):
            if cid is None:
                continue
            ic = int(cid)
            if ic < len(self._class_map) and self._class_map[ic] in self.target_classes:
                keep.append(i)

        return dets[keep][:100] if keep else sv.Detections.empty()