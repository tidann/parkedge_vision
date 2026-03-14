FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System deps for pyzbar
RUN apt-get update && apt-get install -y --no-install-recommends libzbar0 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download EasyOCR models at build time
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

COPY . .

EXPOSE 8000

CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8000"]
