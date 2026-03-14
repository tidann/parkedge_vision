"""
Domain service — extracts French license plates and VINs from raw text.
Ported from parkapp/supabase/functions/_shared/scan.ts
"""

import re
from src.domain.detection import Detection, BBox

VIN_REGEX = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")
LICENSE_PLATE_REGEX = re.compile(r"\b[A-Z]{2}-?\d{3}-?[A-Z]{2}\b")


def extract_vin(text: str) -> str:
    normalized = text.upper().replace("O", "0").replace("I", "1").replace("Q", "0")
    m = VIN_REGEX.search(normalized)
    return m.group(0) if m and len(m.group(0)) == 17 else ""


def extract_license_plate(text: str) -> str:
    m = LICENSE_PLATE_REGEX.search(text.upper())
    if not m:
        return ""
    return m.group(0).replace("-", "").replace(" ", "")


def extract_from_texts(
    texts: list[tuple[str, float]],
    frame_index: int = 0,
    bbox_map: dict[str, list] | None = None,
) -> list[Detection]:
    """Given (text, confidence) pairs from OCR, extract valid VINs and plates."""
    detections = []
    bbox_map = bbox_map or {}

    for text, conf in texts:
        upper = text.upper()

        vin = extract_vin(upper)
        if vin:
            raw_bbox = bbox_map.get(text)
            detections.append(Detection(
                value=vin, kind="vin", confidence=conf,
                frame_index=frame_index,
                bbox=BBox(raw_bbox) if raw_bbox else None,
            ))

        plate = extract_license_plate(upper)
        if plate:
            raw_bbox = bbox_map.get(text)
            detections.append(Detection(
                value=plate, kind="plate", confidence=conf,
                frame_index=frame_index,
                bbox=BBox(raw_bbox) if raw_bbox else None,
            ))

    return detections
