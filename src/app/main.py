"""Main entry point for the vehicle speed tracking application."""

import argparse
import cv2
import logging
import sys
from pathlib import Path


from src.config import VisualConfig, SpeedConfig, RuntimeConfig, DetectorConfig, TrackerConfig
from src.detectors.yolo_detector import YOLODetector
from src.tracking.byte_tracker import ByteTrackerWrapper
from src.pipeline.speed_tracker import SpeedTracker
from src.utils.video_io import open_video, get_frame_size_fps, make_video_writer
from src.common.logging_setup import setup_logging
from src.common.errors import (
    AppError,
    VideoIOError,
    ModelLoadError,
    DetectionError,
    TrackingError,
    ConfigError,
)

logger = logging.getLogger(__name__)


def main():
    """Process video and track vehicle speeds."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Vehicle Speed Tracking")
    parser.add_argument("-i", "--input", required=True, help="Input video path")
    parser.add_argument("-o", "--output", help="Output video path (optional)")
    parser.add_argument("-p", "--pixel-to-meter", type=float, default=0.1,
                      help="Pixel to meter conversion factor")
    parser.add_argument("-t", "--threshold", type=float, default=25.0,
                      help="Speed threshold (km/h)")
    args = parser.parse_args()

    # Initialize configurations
    visual_cfg = VisualConfig()
    speed_cfg = SpeedConfig(speed_threshold_kmh=args.threshold)
    detector_cfg = DetectorConfig()
    tracker_cfg = TrackerConfig()

    # Open video source
    try:
        cap = open_video(args.input)
        width, height, fps = get_frame_size_fps(cap)
        runtime = RuntimeConfig(pixel_to_meter=args.pixel_to_meter, frame_rate=fps)
        writer = make_video_writer(args.output, fps, (width, height)) if args.output else None

        # Initialize pipeline components
        detector = YOLODetector(
            model_path=detector_cfg.model_path,
            conf_threshold=detector_cfg.conf_threshold,
            target_classes=detector_cfg.target_classes,
            warmup=detector_cfg.warmup,
        )

        tracker = ByteTrackerWrapper(
            frame_rate=int(fps),
            track_activation_threshold=tracker_cfg.track_activation_threshold,
            lost_track_seconds=tracker_cfg.lost_track_seconds,
            minimum_matching_threshold=tracker_cfg.minimum_matching_threshold,
        )

        pipeline = SpeedTracker(visual_cfg, speed_cfg, runtime)

        logger.info(
            "Input=%s | Output=%s | %dx%d @ %.2ffps | p2m=%.4f | threshold=%.1f kph",
            args.input, args.output or "<display>", width, height, fps,
            runtime.pixel_to_meter, speed_cfg.speed_threshold_kmh
        )

        frame_idx = 0
        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    # Distinguish between normal end-of-file and a transient read error.
                    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total > 0 and pos >= total:
                        logger.info("End of stream: read frame %d of %d", pos, total)
                    else:
                        logger.error("Frame read error at frame %d of %d", pos, total)
                    break

                try:
                    dets = detector.detect(frame)
                except DetectionError as de:
                    logger.error("Detection error: %s (skipping frame)", de)
                    continue

                try:
                    tracked = tracker.update(dets)
                except TrackingError as te:
                    logger.error("Tracking error: %s (skipping frame)", te)
                    continue

                annotated = pipeline.step(frame, tracked, frame_idx)
                if writer:
                    writer.write(annotated)
                else:
                    cv2.imshow("Vehicle Speed Tracking", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User quit.")
                        break

                frame_idx += 1
                if frame_idx % 100 == 0:
                    logger.info("Processed %d frames", frame_idx)

        except (VideoIOError, ModelLoadError, ConfigError) as e:
            logger.critical("Fatal error: %s", e, exc_info=True)
            sys.exit(2)
        except AppError as e:
            logger.critical("Unhandled app error: %s", e, exc_info=True)
            sys.exit(3)
        finally:
            cap.release()
            if writer:
                writer.release()
            else:
                cv2.destroyAllWindows()

            # Export observations (if pipeline collected any) next to output if provided
            try:
                if hasattr(pipeline, "export_observations"):
                    if args.output:
                        out_csv = Path(args.output).with_suffix('.csv')
                    else:
                        out_csv = Path('data') / 'observations.csv'
                    pipeline.export_observations(str(out_csv))
            except Exception:
                logger.exception("Failed to export observations")

    except Exception as e:
        logger.critical("Initialization error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    setup_logging("INFO")
    main()