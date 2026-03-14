FROM paddlepaddle/paddle:3.0.0-gpu-cuda12.6-cudnn9.5-trt10.7

WORKDIR /app

# System deps for pyzbar
RUN apt-get update && apt-get install -y --no-install-recommends libzbar0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir paddleocr opencv-python-headless fastapi "uvicorn[standard]" websockets numpy pyzbar

# Pre-download OCR models at build time so startup is instant
RUN PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True python -c "\
from paddleocr import PaddleOCR; \
PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, lang='en')"

COPY . .

EXPOSE 8000

CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8000"]
