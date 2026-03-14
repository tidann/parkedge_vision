"""Quick OCR test — saves a frame from the video and runs PaddleOCR on it."""

import cv2
from paddleocr import PaddleOCR

cap = cv2.VideoCapture("videos/sample_short_cropped.mp4")
for _ in range(300):
    cap.read()
ret, frame = cap.read()
cap.release()

print("Frame shape:", frame.shape)
cv2.imwrite("/tmp/test_frame.jpg", frame)

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en",
)

result = ocr.ocr(frame)
print("Result type:", type(result))
print("Truthy:", bool(result))

if result:
    r = result[0]
    print("result[0] type:", type(r))
    if r is None:
        print("result[0] is None")
    elif isinstance(r, dict):
        print("Keys:", r.keys())
        print("rec_texts:", r.get("rec_texts"))
        print("rec_scores:", r.get("rec_scores"))
    elif isinstance(r, list):
        print("result[0] is a list, len:", len(r))
        if r:
            print("First item:", r[0])
    else:
        print("result[0]:", r)
else:
    print("Result is empty/None")
