"""
Video source abstraction — shared frame buffer that any source can write to.
The main app and OCR scanner both read from this buffer.
"""

import threading
import numpy as np


class FrameBuffer:
    """
    Thread-safe single-frame buffer.
    The video source writes the latest frame here.
    The scanner reads it for OCR.
    The MJPEG streamer reads it for browser display.
    """

    def __init__(self):
        self._frame: np.ndarray | None = None
        self._index: int = 0
        self._lock = threading.Lock()
        self._last_read_index: int = -1

    def write(self, frame: np.ndarray, index: int):
        with self._lock:
            self._frame = frame
            self._index = index

    def latest(self) -> tuple[np.ndarray, int] | None:
        """Get the latest frame if it hasn't been read yet."""
        with self._lock:
            if self._frame is None or self._index == self._last_read_index:
                return None
            self._last_read_index = self._index
            return self._frame.copy(), self._index

    def current(self) -> tuple[np.ndarray, int] | None:
        """Get the current frame (even if already read). For MJPEG streaming."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame, self._index

    @property
    def has_frame(self) -> bool:
        with self._lock:
            return self._frame is not None

    @property
    def frame_index(self) -> int:
        with self._lock:
            return self._index
