FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir "fastapi[standard]" uvicorn pillow numpy \
 && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY app /app/app
COPY helper_lib /app/helper_lib
COPY artifacts/model.pt /app/artifacts/model.pt
COPY artifacts/labels.txt /app/artifacts/labels.txt

EXPOSE 80
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

