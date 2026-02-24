from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from config import Config
from detector import resolve_device
from reid import ReIDGallery, extract_embedding


@dataclass
class Track:
    track_id: int  # Re-ID-stable identifier
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float


class TrackerManager:
    """
    Orchestrates YOLOv8n + ByteTrack (via ultralytics) and a Re-ID gallery.

    For every frame the caller passes to update():
      - If it is a detection frame, the YOLO tracker runs, Re-ID remapping is
        applied, and the gallery is updated.
      - If it is a skipped frame, the last known tracks are returned as-is so
        the visualiser always has something to draw.
    """

    def __init__(self, config: Config, model: YOLO, gallery: ReIDGallery) -> None:
        self.config = config
        self.model = model
        self.gallery = gallery
        self.device = resolve_device(config.device)

        # bt_id → stable (Re-ID) id
        self._id_map: dict[int, int] = {}
        # stable id → Track  (last known position for every confirmed track)
        self._last_tracks: dict[int, Track] = {}
        # bt_ids that were active in the previous detection frame
        self._prev_bt_ids: set[int] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray, frame_num: int) -> list[Track]:
        """Return current tracks for *frame*.  Frame skipping is handled here."""
        if frame_num % self.config.skip_frames != 0:
            # Return cached positions on skipped frames
            return list(self._last_tracks.values())
        return self._run_detection(frame, frame_num)

    # ------------------------------------------------------------------
    # Internal: one detection frame
    # ------------------------------------------------------------------

    def _run_detection(self, frame: np.ndarray, frame_num: int) -> list[Track]:
        results = self.model.track(
            frame,
            classes=[0],
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            device=self.device,
        )

        current_bt_ids: set[int] = set()
        raw_tracks: list[tuple[int, int, int, int, int, float]] = []  # bt_id, x1,y1,x2,y2, conf

        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue
            for box, tid in zip(r.boxes, r.boxes.id):
                bt_id = int(tid.item())
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                current_bt_ids.add(bt_id)
                raw_tracks.append((bt_id, x1, y1, x2, y2, conf))

        # Detect tracks that vanished since last detection frame
        vanished = self._prev_bt_ids - current_bt_ids
        for bt_id in vanished:
            stable_id = self._id_map.get(bt_id, bt_id)
            self.gallery.promote_to_lost(stable_id, frame_num)
            self._last_tracks.pop(stable_id, None)

        # Process each current track
        tracks_out: list[Track] = []
        for bt_id, x1, y1, x2, y2, conf in raw_tracks:
            stable_id = self._resolve_id(bt_id, frame, x1, y1, x2, y2, frame_num)

            # Update appearance history
            emb = extract_embedding(frame, x1, y1, x2, y2)
            if emb is not None:
                self.gallery.add_to_gallery(stable_id, emb, frame_num)

            track = Track(track_id=stable_id, x1=x1, y1=y1, x2=x2, y2=y2, conf=conf)
            self._last_tracks[stable_id] = track
            tracks_out.append(track)

        self._prev_bt_ids = current_bt_ids
        return tracks_out

    # ------------------------------------------------------------------
    # ID resolution with Re-ID
    # ------------------------------------------------------------------

    def _resolve_id(
        self,
        bt_id: int,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        frame_num: int,
    ) -> int:
        # Already known → return mapped stable id
        if bt_id in self._id_map:
            return self._id_map[bt_id]

        # Brand-new ByteTrack id: try to match against Re-ID gallery
        emb = extract_embedding(frame, x1, y1, x2, y2)
        matched_id: int | None = None
        if emb is not None:
            matched_id = self.gallery.query(emb, frame_num)

        stable_id = matched_id if matched_id is not None else bt_id
        self._id_map[bt_id] = stable_id
        return stable_id
