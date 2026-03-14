"""
Infrastructure — EasyOCR + pyzbar wrapper.
Knows how to extract text and barcodes from a frame.
Returns domain-level Detection objects.
"""

import logging
import cv2
import numpy as np
import easyocr
from pyzbar import pyzbar

from src.domain.detection import Detection, BBox
from src.domain.extraction import extract_from_texts, extract_vin

logger = logging.getLogger(__name__)


def _detect_barcodes(frame: np.ndarray, frame_index: int) -> list[Detection]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    barcodes = pyzbar.decode(gray)
    detections = []

    for bc in barcodes:
        data = bc.data.decode("utf-8", errors="ignore").strip().upper()
        vin = extract_vin(data)
        if vin:
            r = bc.rect
            bbox = BBox([
                [r.left, r.top],
                [r.left + r.width, r.top],
                [r.left + r.width, r.top + r.height],
                [r.left, r.top + r.height],
            ])
            detections.append(Detection(
                value=vin, kind="vin", confidence=1.0,
                frame_index=frame_index, bbox=bbox,
            ))
            logger.info("Barcode [%s] frame %d: %s", bc.type, frame_index, vin)

    return detections


class PaddleOCREngine:
    """OCR engine using EasyOCR (GPU-accelerated via PyTorch)."""

    def __init__(self, ocr_max_width: int = 1280):
        self._max_width = ocr_max_width
        logger.info("Initializing EasyOCR (max_width=%d)...", ocr_max_width)
        self._reader = easyocr.Reader(["en"], gpu=True)
        logger.info("EasyOCR ready (GPU).")

    def process_frame(
        self, frame: np.ndarray, frame_index: int = 0,
    ) -> tuple[list[Detection], list[tuple[list, str, float]]]:
        """
        Returns (vehicle_detections, all_ocr_lines).
        all_ocr_lines: [(bbox_points, text, confidence), ...] in original frame coords.
        """
        h, w = frame.shape[:2]
        scale = 1.0
        ocr_frame = frame
        if w > self._max_width:
            scale = self._max_width / w
            ocr_frame = cv2.resize(frame, (self._max_width, int(h * scale)))

        detections = list(_detect_barcodes(frame, frame_index))
        all_ocr_lines: list[tuple[list, str, float]] = []

        results = self._reader.readtext(ocr_frame)
        # results: list of (bbox, text, confidence)
        # bbox: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]

        texts: list[tuple[str, float]] = []
        bbox_map: dict[str, list] = {}
        scale_up = 1.0 / scale

        for bbox, text, score in results:
            bbox_orig = [[int(p[0] * scale_up), int(p[1] * scale_up)] for p in bbox]
            texts.append((text, score))
            bbox_map[text] = bbox_orig
            all_ocr_lines.append((bbox_orig, text, score))

        ocr_dets = extract_from_texts(texts, frame_index=frame_index, bbox_map=bbox_map)
        barcode_values = {d.value for d in detections}
        for d in ocr_dets:
            if d.value not in barcode_values:
                detections.append(d)

        return detections, all_ocr_lines
