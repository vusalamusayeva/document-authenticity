FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    tesseract-ocr-eng \
    tesseract-ocr-aze \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.lock ./
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/
COPY models/ ./models/

RUN pip install --no-cache-dir -r requirements.lock \
    && pip install --no-cache-dir --no-deps . \
    && python -c "import cv2, numpy as np; assert cv2.__version__ == '4.10.0.84', cv2.__version__; from doc_auth.deepfake.face_detector import _load_face_cascade, detect_faces; c = _load_face_cascade(); assert not c.empty(); out = detect_faces(np.zeros((160, 160, 3), dtype=np.uint8)); assert isinstance(out, list); print('DOCKER_OPENCV_HAAR_OK', cv2.__version__)"

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000


HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1


RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/tmp_uploads /app/outputs \
    && chown -R appuser:appuser /app
USER appuser

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
