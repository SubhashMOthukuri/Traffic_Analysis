"""Runtime performance benchmark for vehicle speed tracking pipeline."""

from __future__ import annotations
import argparse
import json
import logging
import numpy as np
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from src.common.logging_setup import setup_logging
from ..config import VisualConfig, SpeedConfig, RuntimeConfig, DetectorConfig, TrackerConfig
from ..detectors.yolo_detector import YOLODetector
from ..tracking.byte_tracker import ByteTrackerWrapper
from ..pipeline.speed_tracker import SpeedTracker
from ..utils.video_io import open_video, get_frame_size_fps

logger = logging.getLogger(__name__)


@dataclass
class StreamMetrics:
    """Performance metrics for video stream processing."""
    fps: float
    avg_latency_ms: float
    p95_latency_ms: float
    id_stability: float
    speed_jitter_kmh: float


def percentile(values: List[float], p: float) -> float:
    """Calculate the p-th percentile of a list of values."""
    if not values:
        return 0.0
    k = int(round(p * len(values) / 100))
    return sorted(values)[min(k, len(values)-1)]


def run_benchmark(
    video_path: str,
    out_report: str,
    warmup_frames: int = 30,
    measure_frames: int = 500,
) -> StreamMetrics:
    """Run pipeline benchmark and generate performance report."""
    # Initialize pipeline components
    visual_cfg = VisualConfig()
    speed_cfg = SpeedConfig()
    det_cfg = DetectorConfig()
    trk_cfg = TrackerConfig()
    
    # Open video source
    cap = open_video(video_path)
    width, height, fps = get_frame_size_fps(cap)
    runtime = RuntimeConfig(frame_rate=fps)
    
    det = YOLODetector(
        model_path=det_cfg.model_path,
        conf_threshold=det_cfg.conf_threshold,
        target_classes=det_cfg.target_classes,
        warmup=det_cfg.warmup,
    )
    
    trk = ByteTrackerWrapper(
        frame_rate=int(fps),
        track_activation_threshold=trk_cfg.track_activation_threshold,
        lost_track_seconds=trk_cfg.lost_track_seconds,
        minimum_matching_threshold=trk_cfg.minimum_matching_threshold,
    )
    
    pipe = SpeedTracker(visual_cfg, speed_cfg, runtime)

    # Warmup phase
    logger.info("Warmup phase: %d frames", warmup_frames)
    for _ in range(warmup_frames):
        ok, f = cap.read()
        if not ok:
            break
        _ = pipe.step(f, trk.update(det.detect(f)))

    # Measurement phase
    logger.info("Measurement phase: %d frames", measure_frames)
    latencies: List[float] = []
    id_len: Dict[int, int] = {}
    per_id_speeds: Dict[int, List[float]] = {}
    
    start = time.time()
    frames = 0
    while frames < measure_frames:
        ok, f = cap.read()
        if not ok:
            break
            
        t0 = time.time()
        dets = det.detect(f)
        tracked = trk.update(dets)
        _ = pipe.step(f, tracked)
        dt = (time.time() - t0) * 1000.0
        latencies.append(dt)

        # Collect ID stats
        if getattr(tracked, "tracker_id", None) is not None:
            for tid in tracked.tracker_id:
                tid = int(tid)
                id_len[tid] = id_len.get(tid, 0) + 1
                
            for i in range(len(tracked)):
                tid = int(tracked.tracker_id[i])
                spd = pipe.tracks.get(tid).speed_kmh if tid in pipe.tracks else 0.0
                per_id_speeds.setdefault(tid, []).append(spd)

        frames += 1

    # Calculate metrics
    total_time = time.time() - start
    fps = frames / total_time if total_time > 0 else 0.0
    avg_latency = sum(latencies) / max(1, len(latencies))
    p95 = percentile(latencies, 95.0)

    # ID stability: ids with lifespan >= 10 frames over total ids
    stable_ids = sum(1 for v in id_len.values() if v >= 10)
    stability = stable_ids / max(1, len(id_len))

    # Speed jitter: mean per-id stddev
    jitters = [float(np.std(v)) for v in per_id_speeds.values() if len(v) >= 5]
    speed_jitter = (sum(jitters) / len(jitters)) if jitters else 0.0

    metrics = StreamMetrics(
        fps=fps,
        avg_latency_ms=avg_latency,
        p95_latency_ms=p95,
        id_stability=stability,
        speed_jitter_kmh=speed_jitter,
    )

    Path(out_report).write_text(json.dumps(metrics.__dict__, indent=2))
    logger.info("Benchmark report saved to %s", out_report)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run runtime benchmark (no fine-tuning)")
    ap.add_argument("-i", "--input", required=True, help="Path to input video")
    ap.add_argument("-o", "--out", default="benchmark_report.json", help="Output JSON report path")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--frames", type=int, default=500)
    args = ap.parse_args()
    
    setup_logging("INFO")
    run_benchmark(args.input, args.out, args.warmup, args.frames)