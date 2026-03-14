"""Quick OCR test — tries multiple approaches to find text in a video frame."""

import cv2
import paddle
from paddleocr import PaddleOCR

print("Paddle device:", paddle.device.get_device())
print("CUDA available:", paddle.device.is_compiled_with_cuda())

cap = cv2.VideoCapture("videos/sample_short_cropped.mp4")
for _ in range(300):
    cap.read()
ret, frame = cap.read()
cap.release()

print("Frame shape:", frame.shape)
cv2.imwrite("/tmp/test_frame.jpg", frame)

# Test 1: predict() instead of ocr()
print("\n=== Test 1: predict() ===")
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en",
    text_det_thresh=0.1,
)

for result in ocr.predict("/tmp/test_frame.jpg"):
    print("rec_texts:", result["rec_texts"])
    print("rec_scores:", result["rec_scores"])
    print("dt_polys count:", len(result["dt_polys"]))

# Test 2: predict() with numpy array
print("\n=== Test 2: predict() with numpy array ===")
for result in ocr.predict(frame):
    print("rec_texts:", result["rec_texts"])
    print("rec_scores:", result["rec_scores"])
    print("dt_polys count:", len(result["dt_polys"]))

# Test 3: Try different frames
print("\n=== Test 3: multiple frames ===")
cap = cv2.VideoCapture("videos/sample_short_cropped.mp4")
for i in range(0, 1000, 100):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        break
    for result in ocr.predict(frame):
        texts = result["rec_texts"]
        if texts:
            print(f"Frame {i}: {texts}")
        else:
            print(f"Frame {i}: (no text)")
cap.release()
