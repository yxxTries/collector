from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import Config

# Spatial grid for the Re-ID histogram (rows x cols)
_GRID_ROWS = 4
_GRID_COLS = 2
# HSV histogram bins per cell
_H_BINS = 16
_S_BINS = 8
# Resulting embedding dimensionality: _H_BINS * _S_BINS * _GRID_ROWS * _GRID_COLS
EMBEDDING_DIM = _H_BINS * _S_BINS * _GRID_ROWS * _GRID_COLS  # 16*8*4*2 = 1024


def extract_embedding(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray | None:
    """
    Extract a spatial-pyramid HSV color histogram from a person crop.

    Returns a normalised 1-D numpy array of length EMBEDDING_DIM, or None if
    the crop is invalid (zero area).
    """
    h, w = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    crop = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    cell_h = 128 // _GRID_ROWS
    cell_w = 64 // _GRID_COLS

    parts: list[np.ndarray] = []
    for row in range(_GRID_ROWS):
        for col in range(_GRID_COLS):
            cell = hsv[
                row * cell_h: (row + 1) * cell_h,
                col * cell_w: (col + 1) * cell_w,
            ]
            hist = cv2.calcHist(
                [cell], [0, 1], None,
                [_H_BINS, _S_BINS],
                [0, 180, 0, 256],
            )
            parts.append(hist.flatten())

    embedding = np.concatenate(parts).astype(np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm
    return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity in [0, 1] between two L2-normalised vectors."""
    return float(np.dot(a, b))


@dataclass
class GalleryEntry:
    original_id: int
    embeddings: deque = field(default_factory=deque)
    last_seen_frame: int = 0


class ReIDGallery:
    """
    Maintains appearance embeddings for active and recently-lost tracks.

    Workflow:
      - While a track is active: call add_to_gallery() every frame.
      - When ByteTrack drops a track: call promote_to_lost().
      - When a brand-new ByteTrack ID appears: call query() to check if it
        matches a lost entry; if so, use the returned original_id.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        # Active tracks: bt_id → GalleryEntry  (kept only while track is alive)
        self._active: dict[int, GalleryEntry] = {}
        # Lost tracks: original_id → GalleryEntry
        self._lost: dict[int, GalleryEntry] = {}

    # ------------------------------------------------------------------
    # Active track management
    # ------------------------------------------------------------------

    def add_to_gallery(self, track_id: int, embedding: np.ndarray, frame_num: int) -> None:
        """Update the rolling embedding history for an active track."""
        if track_id not in self._active:
            self._active[track_id] = GalleryEntry(original_id=track_id)
        entry = self._active[track_id]
        entry.embeddings.append(embedding)
        if len(entry.embeddings) > self.config.reid_history_size:
            entry.embeddings.popleft()
        entry.last_seen_frame = frame_num

    def promote_to_lost(self, track_id: int, frame_num: int) -> None:
        """Move an active track to the lost gallery when ByteTrack drops it."""
        if track_id not in self._active:
            return
        entry = self._active.pop(track_id)
        entry.last_seen_frame = frame_num
        # Keep the original_id mapping so that after a remap the lost entry
        # still carries the user-visible identifier.
        self._lost[entry.original_id] = entry

    def remove_active(self, track_id: int) -> None:
        """Completely discard active state (e.g. when no embedding available)."""
        self._active.pop(track_id, None)

    # ------------------------------------------------------------------
    # Re-ID query
    # ------------------------------------------------------------------

    def query(self, embedding: np.ndarray, frame_num: int) -> int | None:
        """
        Compare *embedding* against all unexpired lost-gallery entries.

        Returns the original_id of the best match if similarity exceeds the
        configured threshold, otherwise None.  Expired entries are pruned.
        """
        self._prune(frame_num)
        if not self._lost:
            return None

        best_id: int | None = None
        best_sim: float = self.config.reid_similarity_threshold

        for orig_id, entry in self._lost.items():
            if not entry.embeddings:
                continue
            mean_emb = self._mean_embedding(list(entry.embeddings))
            sim = cosine_similarity(embedding, mean_emb)
            if sim > best_sim:
                best_sim = sim
                best_id = orig_id

        if best_id is not None:
            # Remove from lost so it can't be claimed again
            del self._lost[best_id]

        return best_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune(self, frame_num: int) -> None:
        expired = [
            oid for oid, entry in self._lost.items()
            if frame_num - entry.last_seen_frame > self.config.gallery_ttl_frames
        ]
        for oid in expired:
            del self._lost[oid]

    @staticmethod
    def _mean_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
        mean = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean /= norm
        return mean
