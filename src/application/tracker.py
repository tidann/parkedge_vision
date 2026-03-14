"""
Application service — deduplicates detections across frames.
A number must be seen min_hits times to be confirmed.
"""

import time
from src.domain.detection import Detection, TrackedVehicle


class DetectionTracker:

    def __init__(self, min_hits: int = 2, expiry_seconds: float = 30.0):
        self._pending: dict[str, TrackedVehicle] = {}
        self._confirmed: dict[str, TrackedVehicle] = {}
        self.min_hits = min_hits
        self.expiry_seconds = expiry_seconds

    def update(self, detections: list[Detection]) -> list[TrackedVehicle]:
        now = time.time()
        newly_confirmed = []

        for det in detections:
            key = det.value

            if key in self._confirmed:
                t = self._confirmed[key]
                t.hit_count += 1
                t.last_seen = now
                t.best_confidence = max(t.best_confidence, det.confidence)
                continue

            if key in self._pending:
                t = self._pending[key]
                t.hit_count += 1
                t.last_seen = now
                t.best_confidence = max(t.best_confidence, det.confidence)
                if t.hit_count >= self.min_hits:
                    self._confirmed[key] = t
                    del self._pending[key]
                    newly_confirmed.append(t)
            else:
                self._pending[key] = TrackedVehicle(
                    value=det.value, kind=det.kind,
                    best_confidence=det.confidence,
                    first_seen=now, last_seen=now,
                )

        expired = [k for k, v in self._pending.items() if now - v.last_seen > self.expiry_seconds]
        for k in expired:
            del self._pending[k]

        return newly_confirmed

    @property
    def confirmed(self) -> list[TrackedVehicle]:
        return list(self._confirmed.values())

    @property
    def pending(self) -> list[TrackedVehicle]:
        return list(self._pending.values())

    def reset(self):
        self._pending.clear()
        self._confirmed.clear()
