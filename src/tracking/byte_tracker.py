"""ByteTrack object tracker wrapper with error handling."""

from __future__ import annotations
from dataclasses import dataclass
import supervision as sv
from src.common.errors import TrackingError


@dataclass
class ByteTrackerWrapper:
    """Wrapper for ByteTrack multi-object tracker."""
    frame_rate: int
    track_activation_threshold: float
    lost_track_seconds: float
    minimum_matching_threshold: float

    def __post_init__(self) -> None:
        """Initialize ByteTrack with configuration parameters."""
        self.tracker = sv.ByteTrack(
            frame_rate=self.frame_rate,
            track_activation_threshold=self.track_activation_threshold,
            lost_track_buffer=int(self.frame_rate * self.lost_track_seconds),
            minimum_matching_threshold=self.minimum_matching_threshold,
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Update tracker state with new detections."""
        try:
            return self.tracker.update_with_detections(detections)
        except Exception as e:
            raise TrackingError(f"Tracker update failed: {e}") from e