from __future__ import annotations

from collections import defaultdict, deque

import cv2
import numpy as np

from config import Config
from tracker import Track


class Visualizer:
    """
    Draws bounding boxes, track-ID labels, and trail dots onto copies of
    video frames.  All rendering is done on CPU with OpenCV.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        # track_id → deque of (cx, cy) trail points
        self._trails: dict[int, deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=config.trail_length)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw(self, frame: np.ndarray, tracks: list[Track], frame_num: int) -> np.ndarray:
        """Return an annotated copy of *frame* (original is unchanged)."""
        canvas = frame.copy()

        for track in tracks:
            color = self._get_color(track.track_id)
            cx = (track.x1 + track.x2) // 2
            cy = (track.y1 + track.y2) // 2
            self._trails[track.track_id].append((cx, cy))

            self._draw_box(canvas, track, color)
            self._draw_label(canvas, track, color)

        # Draw trails for all tracks that still have history
        active_ids = {t.track_id for t in tracks}
        for tid, trail in self._trails.items():
            color = self._get_color(tid)
            self._draw_trail(canvas, trail, color, highlight=(tid in active_ids))

        self._draw_hud(canvas, len(tracks), frame_num)
        return canvas

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_box(self, canvas: np.ndarray, track: Track, color: tuple[int, int, int]) -> None:
        cv2.rectangle(canvas, (track.x1, track.y1), (track.x2, track.y2), color, 2)

    def _draw_label(self, canvas: np.ndarray, track: Track, color: tuple[int, int, int]) -> None:
        label = f"ID {track.track_id}  {track.conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        lx = track.x1
        ly = max(track.y1 - 4, th + 4)
        # Background rectangle for readability
        cv2.rectangle(canvas, (lx, ly - th - baseline), (lx + tw, ly + baseline), color, cv2.FILLED)
        cv2.putText(
            canvas, label, (lx, ly),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
        )

    def _draw_trail(
        self,
        canvas: np.ndarray,
        trail: deque[tuple[int, int]],
        color: tuple[int, int, int],
        highlight: bool,
    ) -> None:
        pts = list(trail)
        for i, (px, py) in enumerate(pts):
            # Older points are smaller and dimmer
            alpha = (i + 1) / len(pts)
            faded = tuple(int(c * alpha) for c in color)
            radius = 2 if i < len(pts) - 1 else 4
            cv2.circle(canvas, (px, py), radius, faded, cv2.FILLED)

    def _draw_hud(self, canvas: np.ndarray, num_tracks: int, frame_num: int) -> None:
        text = f"Frame {frame_num}  |  Tracks: {num_tracks}"
        cv2.putText(
            canvas, text, (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, text, (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # Color utility
    # ------------------------------------------------------------------

    @staticmethod
    def _get_color(track_id: int) -> tuple[int, int, int]:
        """Return a deterministic BGR color for a given track ID."""
        hue = (track_id * 37) % 180
        hsv_pixel = np.uint8([[[hue, 220, 200]]])
        bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])
