"""Quick OCR test — tries multiple model configs."""

import os
import shutil
import cv2
import paddle

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

print("Paddle device:", paddle.device.get_device())
print("CUDA:", paddle.device.is_compiled_with_cuda())

cap = cv2.VideoCapture("videos/sample_short_cropped.mp4")
for _ in range(300):
    cap.read()
ret, frame = cap.read()
cap.release()
print("Frame shape:", frame.shape)
cv2.imwrite("/tmp/test_frame.jpg", frame)

# Delete cached models to force re-download
model_dir = os.path.expanduser("~/.paddlex/official_models")
if os.path.exists(model_dir):
    print("Deleting cached models...")
    shutil.rmtree(model_dir)

from paddleocr import PaddleOCR

# Test 1: Default (PP-OCRv5)
print("\n=== Test 1: Default PP-OCRv5 ===")
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en",
)
for r in ocr.predict("/tmp/test_frame.jpg"):
    print("texts:", r["rec_texts"][:5] if r["rec_texts"] else "(empty)")

# Test 2: PP-OCRv4
print("\n=== Test 2: PP-OCRv4 ===")
ocr2 = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en",
    text_det_model_name="PP-OCRv4_server_det",
    text_rec_model_name="en_PP-OCRv4_rec",
)
for r in ocr2.predict("/tmp/test_frame.jpg"):
    print("texts:", r["rec_texts"][:5] if r["rec_texts"] else "(empty)")

# Test 3: mobile det model
print("\n=== Test 3: PP-OCRv5 mobile det ===")
ocr3 = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en",
    text_det_model_name="PP-OCRv5_mobile_det",
)
for r in ocr3.predict("/tmp/test_frame.jpg"):
    print("texts:", r["rec_texts"][:5] if r["rec_texts"] else "(empty)")

# Test 4: Force CPU to compare
print("\n=== Test 4: Force CPU ===")
paddle.set_device("cpu")
ocr4 = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en",
)
for r in ocr4.predict("/tmp/test_frame.jpg"):
    print("texts:", r["rec_texts"][:5] if r["rec_texts"] else "(empty)")
