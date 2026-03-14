"""
Application service — orchestrates the scanning pipeline.
Pulls frames from a shared frame buffer, runs OCR, publishes results.
"""

import logging
import threading
import time
from typing import Callable

import numpy as np

from src.domain.detection import Detection, FrameResult
from src.application.tracker import DetectionTracker
from src.infrastructure.ocr.paddle_ocr import PaddleOCREngine
from src.infrastructure.video.source import FrameBuffer

logger = logging.getLogger(__name__)

OnResultCallback = Callable[[FrameResult, list], None]  # (result, newly_confirmed)


class ScannerService:
    """Reads frames from the shared buffer, runs OCR, tracks detections."""

    def __init__(
        self,
        frame_buffer: FrameBuffer,
        ocr_engine: PaddleOCREngine,
        tracker: DetectionTracker,
        scan_interval_ms: int = 500,
    ):
        self._buffer = frame_buffer
        self._ocr = ocr_engine
        self._tracker = tracker
        self._scan_interval = scan_interval_ms / 1000.0
        self._running = False
        self._thread: threading.Thread | None = None
        self._callbacks: list[OnResultCallback] = []

    def on_result(self, cb: OnResultCallback):
        self._callbacks.append(cb)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Scanner started (interval=%.1fs)", self._scan_interval)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Scanner stopped")

    @property
    def running(self) -> bool:
        return self._running

    def _loop(self):
        while self._running:
            frame_data = self._buffer.latest()
            if frame_data is None:
                time.sleep(0.05)
                continue

            frame, frame_index = frame_data
            detections, ocr_lines = self._ocr.process_frame(frame, frame_index)
            newly_confirmed = self._tracker.update(detections)

            result = FrameResult(
                frame_index=frame_index,
                detections=detections,
                ocr_lines=ocr_lines,
            )

            for cb in self._callbacks:
                try:
                    cb(result, newly_confirmed)
                except Exception as e:
                    logger.error("Callback error: %s", e)

            time.sleep(self._scan_interval)
