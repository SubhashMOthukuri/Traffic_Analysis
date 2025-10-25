"""Speed tracking pipeline with visualization."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Deque, Tuple
from collections import deque
import numpy as np
import time
import cv2
import logging
import csv

from ..config import VisualConfig, SpeedConfig, RuntimeConfig
from ..utils.geometry import midpoint, pixel_distance

logger = logging.getLogger(__name__)


@dataclass
class TrackState:
    """State of a tracked object including speed history."""
    history: Deque[Tuple[Tuple[int, int], float]] = field(default_factory=lambda: deque(maxlen=5))
    speed_kmh: float = 0.0
    velocity_mps: np.ndarray = field(default_factory=lambda: np.zeros(2))


class SpeedTracker:
    """Pipeline for tracking and computing vehicle speeds."""
    
    def __init__(self, visual_cfg: VisualConfig, speed_cfg: SpeedConfig, runtime_cfg: RuntimeConfig):
        """Initialize tracker with configuration parameters."""
        self.visual = visual_cfg
        self.speed_cfg = speed_cfg
        self.runtime = runtime_cfg
        self.tracks: Dict[int, TrackState] = {}
        # observations: list of tuples (frame_idx, track_id, speed_kmh, cx, cy)
        self.observations = []

    def _smooth(self, old: float, new: float) -> float:
        """Apply EMA smoothing to scalar values."""
        return old * self.speed_cfg.smoothing_ema + new * (1 - self.speed_cfg.smoothing_ema)

    def _smooth_vec(self, old: np.ndarray, new: np.ndarray) -> np.ndarray:
        """Apply EMA smoothing to vector values."""
        return old * self.speed_cfg.smoothing_ema + new * (1 - self.speed_cfg.smoothing_ema)

    def _compute_speed(self, state: TrackState) -> Tuple[float, np.ndarray]:
        """Compute smoothed speed and velocity from position history."""
        if len(state.history) < 2:
            return 0.0, np.zeros(2)

        p1, t1 = state.history[-2]
        p2, t2 = state.history[-1]
        dt = t2 - t1

        # If the measured dt is too small (e.g. because we used wall-clock and processing is
        # faster than frame interval), fall back to the configured frame-rate estimate so
        # realistic speeds can still be computed for fixed-framerate video.
        if dt < self.speed_cfg.min_dt_s_for_speed:
            if self.runtime.frame_rate and self.runtime.frame_rate > 0:
                est_dt = 1.0 / float(self.runtime.frame_rate)
                # If the estimated dt is still below the min threshold, honor min_dt
                dt = max(est_dt, self.speed_cfg.min_dt_s_for_speed)
                logger.debug("Using estimated dt=%.4fs (frame_rate=%.2f) for speed computation", dt, self.runtime.frame_rate)
            else:
                return state.speed_kmh, state.velocity_mps

        dist_px = pixel_distance(p1, p2)
        if dist_px < self.speed_cfg.min_dist_px_for_speed:
            return state.speed_kmh, state.velocity_mps

        dist_m = dist_px * self.runtime.pixel_to_meter
        speed_mps = dist_m / dt
        raw_speed_kmh = speed_mps * 3.6
        raw_v = np.array([
            (p2[0] - p1[0]) * self.runtime.pixel_to_meter / dt,
            (p2[1] - p1[1]) * self.runtime.pixel_to_meter / dt,
        ], dtype=float)

        sm_speed = self._smooth(state.speed_kmh, raw_speed_kmh)
        sm_vec = self._smooth_vec(state.velocity_mps, raw_v)
        return sm_speed, sm_vec

    def step(self, frame, tracked_dets, frame_idx: int | None = None) -> np.ndarray:
        """Process a frame of detections and draw speed annotations."""
        start = time.time()
        now = start
        out = frame.copy()
        present_ids = set()

        try:
            # Guard: cap number of objects per frame
            n = len(tracked_dets) if hasattr(tracked_dets, "__len__") else 0
            if n > self.runtime.max_objects_per_frame:
                logger.warning("Capping objects per frame: %d -> %d", n, self.runtime.max_objects_per_frame)
                tracked_dets = tracked_dets[: self.runtime.max_objects_per_frame]

            if getattr(tracked_dets, "tracker_id", None) is not None and len(tracked_dets.tracker_id) == len(tracked_dets):
                for i in range(len(tracked_dets)):
                    track_id = int(tracked_dets.tracker_id[i])
                    box = tuple(map(int, tracked_dets.xyxy[i]))  # type: ignore
                    present_ids.add(track_id)

                    state = self.tracks.setdefault(track_id, TrackState(history=deque(maxlen=self.speed_cfg.max_trajectory_len)))
                    state.history.append((midpoint(box), now))

                    spd_kmh, v_mps = self._compute_speed(state)
                    state.speed_kmh = spd_kmh
                    state.velocity_mps = v_mps

                    color = self.visual.speeding_color if spd_kmh > self.speed_cfg.speed_threshold_kmh else self.visual.default_color
                    x1, y1, x2, y2 = box

                    # record an observation for export/analysis
                    if frame_idx is not None:
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        self.observations.append((int(frame_idx), int(track_id), float(spd_kmh), int(cx), int(cy)))
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, self.visual.line_thickness)

                    label = f"ID:{track_id} {spd_kmh:.1f} kph"
                    (tw, th), _ = cv2.getTextSize(label, self.visual.font, self.visual.font_scale, 1)
                    tb_y1 = max(0, y1 - th - 5)
                    tb_y2 = y1
                    tb_x1 = max(0, x1)
                    tb_x2 = min(out.shape[1], x1 + tw + 6)
                    if tb_x1 < tb_x2 and tb_y1 < tb_y2:
                        cv2.rectangle(out, (tb_x1, tb_y1), (tb_x2, tb_y2), color, -1)
                        cv2.putText(out, label, (tb_x1 + 2, tb_y2 - 2), self.visual.font, 
                                  self.visual.font_scale, self.visual.text_color, 1, cv2.LINE_AA)

            # Watchdog: per-frame budget
            if (time.time() - start) > self.runtime.per_frame_timeout_s and self.runtime.drop_frame_on_overload:
                logger.warning("Frame processing exceeded %.3fs, dropping rest of objs for this frame",
                             self.runtime.per_frame_timeout_s)
                return out

        finally:
            # Cleanup tracks not present this frame
            stale = [tid for tid in self.tracks.keys() if tid not in present_ids]
            for tid in stale:
                self.tracks.pop(tid, None)

        return out

    def export_observations(self, path: str) -> None:
        """Write collected observations to a CSV file.

        CSV columns: frame_idx,track_id,speed_kmh,center_x,center_y
        """
        if not self.observations:
            logger.info("No observations to export to %s", path)
            return
        try:
            with open(path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["frame_idx", "track_id", "speed_kmh", "center_x", "center_y"])
                for row in self.observations:
                    writer.writerow(row)
            logger.info("Exported %d observations to %s", len(self.observations), path)
        except Exception as e:
            logger.error("Failed to export observations to %s: %s", path, e)