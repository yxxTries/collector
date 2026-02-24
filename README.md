# collector

CCTV-style person tracking with **YOLOv8n + ByteTrack** and a lightweight appearance **Re-ID** module to keep IDs stable across short occlusions / dropouts.

## What it does

- Detects **people only** (COCO class `0`) using Ultralytics YOLO (`yolov8n.pt`).
- Tracks detections with **ByteTrack** (via `ultralytics`’ `model.track(...)`).
- When ByteTrack drops and later re-creates a track, attempts to **re-associate** it using a simple HSV spatial-histogram embedding + cosine similarity.
- Optionally renders a live OpenCV window (boxes, ID labels, trails) and/or exports tracks to CSV.

## To - do
produce dashboard
produce analytics
figure out data storage part
