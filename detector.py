from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO

from config import Config


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float


def resolve_device(device: str) -> str:
    """Return a concrete device string, resolving 'auto' to 'cuda' or 'cpu'."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class Detector:
    """Wraps YOLOv8n for efficient single-class (person) detection."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.model = YOLO(config.model_name)
        self.model.to(self.device)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on *frame* and return person detections."""
        resized, scale_x, scale_y = self._resize(frame)

        results = self.model(
            resized,
            classes=[0],  # person only
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            verbose=False,
            device=self.device,
        )

        detections: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                # Scale back to original frame coordinates
                detections.append(Detection(
                    x1=int(x1 / scale_x),
                    y1=int(y1 / scale_y),
                    x2=int(x2 / scale_x),
                    y2=int(y2 / scale_y),
                    conf=conf,
                ))
        return detections

    def _resize(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Resize frame so width == input_width, preserving aspect ratio."""
        h, w = frame.shape[:2]
        if w <= self.config.input_width:
            return frame, 1.0, 1.0
        scale = self.config.input_width / w
        new_w = self.config.input_width
        new_h = int(h * scale)
        import cv2
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized, scale, scale
