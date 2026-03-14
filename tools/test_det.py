"""Test the text detection model directly."""

import os
import shutil

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import cv2
import numpy as np
import paddle

paddle.set_device("cpu")

# Create synthetic image
img = np.ones((200, 400, 3), dtype=np.uint8) * 255
cv2.putText(img, "HELLO123", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 5)
cv2.imwrite("/tmp/hello.jpg", img)

# Delete cached models
model_dir = os.path.expanduser("~/.paddlex/official_models")
if os.path.exists(model_dir):
    print("Deleting cached models...")
    shutil.rmtree(model_dir)

# Test detection model directly
from paddlex import create_model

print("=== Testing detection model ===")
det_model = create_model("PP-OCRv5_server_det")
for r in det_model.predict("/tmp/hello.jpg"):
    print("polys:", r["dt_polys"])
    print("scores:", r["dt_scores"])

# Also test with the video frame
print("\n=== Testing with video frame ===")
cap = cv2.VideoCapture("videos/sample_short_cropped.mp4")
for _ in range(300):
    cap.read()
ret, frame = cap.read()
cap.release()
cv2.imwrite("/tmp/video_frame.jpg", frame)

for r in det_model.predict("/tmp/video_frame.jpg"):
    print("polys:", len(r["dt_polys"]))
    print("scores:", r["dt_scores"])
