# Image Classifier API (CNN)

## Train
uv run python train_cnn.py

## Run
uv run uvicorn app.main:app --reload --port 8000
# docs: http://127.0.0.1:8000/docs

## Test
curl -X POST "http://127.0.0.1:8000/classify" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/absolute/path/to/img.jpg"

## Docker
```bash
docker build --platform=linux/amd64 -t cnn-api .
docker run --rm -p 8000:80 cnn-api
# docs: http://127.0.0.1:8000/docs

