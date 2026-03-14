"""Test OCR with a synthetic image to verify the installation works."""

import os
import cv2
import numpy as np
import paddle

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
paddle.set_device("cpu")

from paddleocr import PaddleOCR

# Create a simple image with clear text
img = np.ones((200, 400, 3), dtype=np.uint8) * 255
cv2.putText(img, "HELLO123", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 5)
cv2.imwrite("/tmp/hello.jpg", img)

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en",
)

print("=== Synthetic image test ===")
for r in ocr.predict("/tmp/hello.jpg"):
    print("texts:", r["rec_texts"])
    print("scores:", r["rec_scores"])
    print("polys:", len(r["dt_polys"]))

print("\n=== Video frame test ===")
for r in ocr.predict("videos/sample_short_cropped.mp4"):
    if r["rec_texts"]:
        print("texts:", r["rec_texts"])
        break
else:
    print("No text found in any frame")
