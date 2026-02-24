from __future__ import annotations

import csv
from pathlib import Path

from tracker import Track


class TrackExporter:
    """
    Buffers track observations in memory and writes them to a CSV file on request.

    CSV columns: frame, track_id, x1, y1, x2, y2, conf
    """

    _HEADER = ["frame", "track_id", "x1", "y1", "x2", "y2", "conf"]

    def __init__(self, output_path: str) -> None:
        self.output_path = output_path
        self._rows: list[list] = []

    def record(self, frame_num: int, tracks: list[Track]) -> None:
        """Append one row per track for the given frame number."""
        for t in tracks:
            self._rows.append([
                frame_num,
                t.track_id,
                t.x1,
                t.y1,
                t.x2,
                t.y2,
                round(t.conf, 4),
            ])

    def save(self) -> None:
        """Flush all buffered data to the output CSV file."""
        if not self.output_path:
            return
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(self._HEADER)
            writer.writerows(self._rows)
        print(f"[exporter] Saved {len(self._rows)} rows → {path.resolve()}")

    def row_count(self) -> int:
        return len(self._rows)
