"""
Infrastructure — docTR + pyzbar wrapper.
Knows how to extract text and barcodes from a frame.
Returns domain-level Detection objects.
"""

import os
os.environ["USE_TORCH"] = "1"

import logging
import cv2
import numpy as np
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
    """OCR engine using docTR (GPU-accelerated via PyTorch)."""

    def __init__(self, ocr_max_width: int = 1280):
        self._max_width = ocr_max_width
        logger.info("Initializing docTR (max_width=%d)...", ocr_max_width)

        import torch
        from doctr.models import ocr_predictor

        self._predictor = ocr_predictor(
            det_arch="db_mobilenet_v3_large",
            reco_arch="crnn_mobilenet_v3_large",
            pretrained=True,
        )
        if torch.cuda.is_available():
            self._predictor = self._predictor.cuda().half()
            logger.info("docTR ready (GPU FP16: %s)", torch.cuda.get_device_name(0))
        else:
            logger.info("docTR ready (CPU)")

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

        # docTR expects RGB
        rgb = cv2.cvtColor(ocr_frame, cv2.COLOR_BGR2RGB)
        result = self._predictor([rgb])

        oh, ow = ocr_frame.shape[:2]
        scale_up = 1.0 / scale
        texts: list[tuple[str, float]] = []
        bbox_map: dict[str, list] = {}

        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        text = word.value
                        conf = word.confidence
                        # docTR returns relative coords (0-1), convert to pixel
                        (x1r, y1r), (x2r, y2r) = word.geometry
                        x1 = int(x1r * ow * scale_up)
                        y1 = int(y1r * oh * scale_up)
                        x2 = int(x2r * ow * scale_up)
                        y2 = int(y2r * oh * scale_up)
                        bbox_orig = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

                        texts.append((text, conf))
                        bbox_map[text] = bbox_orig
                        all_ocr_lines.append((bbox_orig, text, conf))

        ocr_dets = extract_from_texts(texts, frame_index=frame_index, bbox_map=bbox_map)
        barcode_values = {d.value for d in detections}
        for d in ocr_dets:
            if d.value not in barcode_values:
                detections.append(d)

        return detections, all_ocr_lines
