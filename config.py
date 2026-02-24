from dataclasses import dataclass, field


@dataclass
class Config:
    # Detection
    model_name: str = "yolov8n.pt"
    conf_threshold: float = 0.30
    iou_threshold: float = 0.45
    input_width: int = 640  # resize frame width before inference

    # Tracking (ByteTrack via ultralytics)
    track_max_age: int = 30  # frames before a lost track is dropped

    # Re-ID gallery
    reid_similarity_threshold: float = 0.55  # cosine similarity to claim a match
    gallery_ttl_frames: int = 300            # keep lost tracks for this many frames
    reid_history_size: int = 5              # rolling embeddings to average per track

    # Processing
    skip_frames: int = 1   # run detection every N frames (1 = every frame)
    device: str = "auto"   # "auto" | "cuda" | "cpu"

    # Output
    show: bool = True
    output_csv: str = ""   # path to CSV output; empty string = no export
    trail_length: int = 30  # number of past centers to draw as trail
