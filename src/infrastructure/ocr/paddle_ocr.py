"""
Infrastructure — PaddleOCR + pyzbar wrapper.
Knows how to extract text and barcodes from a frame.
Returns domain-level Detection objects.
"""

import logging
import cv2
import numpy as np
from paddleocr import PaddleOCR
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

    def __init__(self, ocr_max_width: int = 1280):
        self._max_width = ocr_max_width
        logger.info("Initializing PaddleOCR (max_width=%d)...", ocr_max_width)
        self._ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
            text_det_thresh=0.3,
            text_recognition_batch_size=6,
        )
        logger.info("PaddleOCR ready.")

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

        result = self._ocr.ocr(ocr_frame)
        if result and result[0]:
            r = result[0]
            rec_texts = r["rec_texts"]
            rec_scores = r["rec_scores"]
            dt_polys = r["dt_polys"]

            texts: list[tuple[str, float]] = []
            bbox_map: dict[str, list] = {}
            scale_up = 1.0 / scale

            for text, score, poly in zip(rec_texts, rec_scores, dt_polys):
                bbox = poly.tolist() if hasattr(poly, "tolist") else poly
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
