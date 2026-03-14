"""
Domain entities for vehicle number detection.
"""

from dataclasses import dataclass, field
import time


@dataclass(frozen=True)
class BBox:
    """Polygon bounding box — list of [x, y] points."""
    points: list[list[int]]

    def to_list(self) -> list[list[int]]:
        return self.points


@dataclass(frozen=True)
class Detection:
    """A single detected vehicle number in one frame."""
    value: str
    kind: str  # "vin" or "plate"
    confidence: float
    frame_index: int
    bbox: BBox | None = None


@dataclass
class TrackedVehicle:
    """A vehicle number confirmed across multiple frames."""
    value: str
    kind: str
    best_confidence: float
    hit_count: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "kind": self.kind,
            "confidence": round(self.best_confidence, 3),
            "hits": self.hit_count,
            "firstSeen": self.first_seen,
            "lastSeen": self.last_seen,
        }


@dataclass
class FrameResult:
    """OCR result for a single frame — sent to the UI for overlay drawing."""
    frame_index: int
    detections: list[Detection]
    ocr_lines: list[tuple[list[list[int]], str, float]]  # (bbox, text, confidence)
    frame_width: int = 0
    frame_height: int = 0
