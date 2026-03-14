"""Quick OCR test — GPU vs CPU comparison."""

import os
import cv2
import paddle

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

print("Paddle device:", paddle.device.get_device())

cap = cv2.VideoCapture("videos/sample_short_cropped.mp4")
for _ in range(300):
    cap.read()
ret, frame = cap.read()
cap.release()
print("Frame shape:", frame.shape)
cv2.imwrite("/tmp/test_frame.jpg", frame)

from paddleocr import PaddleOCR

# Test 1: GPU
print("\n=== GPU ===")
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en",
)
for r in ocr.predict("/tmp/test_frame.jpg"):
    print("texts:", r["rec_texts"][:5] if r["rec_texts"] else "(empty)")
    print("dt_polys:", len(r["dt_polys"]))

# Test 2: Force CPU
print("\n=== CPU ===")
paddle.set_device("cpu")
ocr2 = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en",
)
for r in ocr2.predict("/tmp/test_frame.jpg"):
    print("texts:", r["rec_texts"][:5] if r["rec_texts"] else "(empty)")
    print("dt_polys:", len(r["dt_polys"]))
