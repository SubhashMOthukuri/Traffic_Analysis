"""Configuration classes for the vehicle speed tracking system."""

from dataclasses import dataclass
from typing import Tuple, Sequence


@dataclass(frozen=True)
class VisualConfig:
    """Visual display configuration for annotations."""
    default_color: Tuple[int, int, int] = (0, 255, 0)
    speeding_color: Tuple[int, int, int] = (0, 0, 255)
    text_color: Tuple[int, int, int] = (255, 255, 255)
    font: int = 0  # cv2.FONT_HERSHEY_SIMPLEX, bound at runtime
    font_scale: float = 0.5
    line_thickness: int = 2


@dataclass(frozen=True)
class SpeedConfig:
    """Speed calculation and filtering configuration."""
    speed_threshold_kmh: float = 25.0
    max_trajectory_len: int = 5
    smoothing_ema: float = 0.6
    min_dist_px_for_speed: float = 2.0
    min_dt_s_for_speed: float = 0.05


@dataclass(frozen=True)
class DetectorConfig:
    """Object detector configuration."""
    model_path: str = "yolov8n.pt"
    conf_threshold: float = 0.4
    target_classes: Sequence[str] = ("car", "truck", "bus", "motorcycle", "bicycle")
    warmup: bool = True


@dataclass(frozen=True)
class TrackerConfig:
    """Object tracking configuration."""
    track_activation_threshold: float = 0.3
    lost_track_seconds: float = 1.5
    minimum_matching_threshold: float = 0.9


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime performance configuration."""
    pixel_to_meter: float = 0.1
    frame_rate: float = 30.0
    max_objects_per_frame: int = 50  # guard: cap work per frame
    per_frame_timeout_s: float = 0.050  # watchdog: 50ms budget/frame for processing
    drop_frame_on_overload: bool = True