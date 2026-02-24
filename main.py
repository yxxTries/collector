"""
CCTV Person Tracker with Re-ID
================================
Usage:
    python main.py --input video.mp4
    python main.py --input video.mp4 --output-csv tracks.csv --device cuda
    python main.py --input video.mp4 --skip-frames 2 --conf 0.35 --no-show

Press  q  or  Esc  in the display window to stop early.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
from ultralytics import YOLO

from config import Config
from detector import resolve_device
from exporter import TrackExporter
from reid import ReIDGallery
from tracker import TrackerManager
from visualizer import Visualizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CCTV person tracker with Re-ID (YOLOv8n + ByteTrack)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input MP4 video")
    parser.add_argument("-o", "--output-csv", default="", help="Path to output CSV file")
    parser.add_argument("--show", dest="show", action="store_true", default=True,
                        help="Show live display window")
    parser.add_argument("--no-show", dest="show", action="store_false",
                        help="Disable live display window")
    parser.add_argument("--skip-frames", type=int, default=1,
                        help="Run detection every N frames (1 = every frame)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="Compute device")
    parser.add_argument("--conf", type=float, default=0.30,
                        help="Detection confidence threshold")
    parser.add_argument("--reid-threshold", type=float, default=0.55,
                        help="Re-ID cosine similarity threshold")
    parser.add_argument("--window-size", default="1280x720",
                        help="Initial display window size as WxH (e.g. 1920x1080)")
    parser.add_argument("--trail-length", type=int, default=30,
                        help="Number of past positions to draw as trail")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_window_size(value: str) -> tuple[int, int]:
    """Parse a 'WxH' string into (width, height) ints."""
    try:
        w, h = value.lower().split("x")
        return int(w), int(h)
    except ValueError:
        print(f"[warn] Invalid --window-size '{value}', using 1280x720")
        return 1280, 720


def main() -> None:
    args = parse_args()

    # Build config from CLI args
    config = Config(
        conf_threshold=args.conf,
        skip_frames=args.skip_frames,
        device=args.device,
        show=args.show,
        output_csv=args.output_csv,
        reid_similarity_threshold=args.reid_threshold,
        trail_length=args.trail_length,
    )

    device = resolve_device(config.device)
    print(f"[main] Device: {device}")
    print(f"[main] Opening: {args.input}")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"[error] Cannot open video: {args.input}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"[main] Video: {total_frames} frames @ {fps_video:.1f} fps")

    # Initialise components
    print("[main] Loading YOLOv8n model …")
    model = YOLO(config.model_name)
    model.to(device)

    gallery = ReIDGallery(config)
    tracker = TrackerManager(config, model, gallery)
    visualizer = Visualizer(config)
    exporter = TrackExporter(config.output_csv)

    all_ids_seen: set[int] = set()
    frame_num = 0

    # FPS measurement
    t_start = time.perf_counter()
    fps_display = 0.0
    fps_accum = 0
    fps_timer = time.perf_counter()

    if config.show:
        win_w, win_h = _parse_window_size(args.window_size)
        cv2.namedWindow("CCTV Tracker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("CCTV Tracker", win_w, win_h)

    print("[main] Running — press  q / Esc  in the window to quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            tracks = tracker.update(frame, frame_num)
            all_ids_seen.update(t.track_id for t in tracks)

            if config.output_csv:
                exporter.record(frame_num, tracks)

            if config.show:
                annotated = visualizer.draw(frame, tracks, frame_num)
                # Overlay real-time FPS
                cv2.putText(
                    annotated,
                    f"FPS {fps_display:.1f}",
                    (8, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
                )
                cv2.imshow("CCTV Tracker", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or Esc
                    print("[main] User requested stop.")
                    break

            # FPS counter (updated every 30 frames)
            fps_accum += 1
            if fps_accum >= 30:
                now = time.perf_counter()
                fps_display = fps_accum / (now - fps_timer)
                fps_timer = now
                fps_accum = 0
                pct = (frame_num / total_frames * 100) if total_frames > 0 else 0
                print(f"  frame {frame_num:>6}  |  tracks {len(tracks):>3}  |  fps {fps_display:>5.1f}  |  {pct:.1f}%")

            frame_num += 1

    finally:
        cap.release()
        if config.show:
            cv2.destroyAllWindows()

    # Summary
    elapsed = time.perf_counter() - t_start
    avg_fps = frame_num / elapsed if elapsed > 0 else 0.0
    print(f"\n[main] Finished — {frame_num} frames in {elapsed:.1f}s  ({avg_fps:.1f} avg fps)")
    print(f"[main] Unique person IDs seen: {len(all_ids_seen)}")

    if config.output_csv:
        exporter.save()


if __name__ == "__main__":
    main()
