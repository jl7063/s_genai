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

## Module 6 Assignment – DCGAN on MNIST + FastAPI Endpoint

Module 6 – DCGAN on MNIST + FastAPI
Train: uv run python train_gan_mnist.py
Export weights for API: cp artifacts_gan/G.pt artifacts/gan_generator.pt
Run API: uv run uvicorn app.main:app --reload --port 8001
Test GAN endpoint: open http://127.0.0.1:8001/docs → POST /generate_gan (e.g., {"num_samples":16})
Evidence: artifacts_gan/epoch_*.png, gan_grid_trained.png.

- **Part 1**: Implemented DCGAN for MNIST in `helper_lib/model.py` (Generator + Discriminator), trainer loop in `helper_lib/trainer.py`, and sampling util in `helper_lib/generator.py`.
- **Part 2**: Trained on MNIST (`train_gan_mnist.py`) and added `/generate_gan` endpoint in `app/main.py`.  
  - Trained weights saved to `artifacts_gan/G.pt` and copied to `artifacts/gan_generator.pt` for the API.
  - Run: `uv run uvicorn app.main:app --reload` and test at `http://127.0.0.1:8000/docs`.
- **Evidence**: See sample grids in `artifacts_gan/epoch_*.png` and `gan_grid.png` (generated via API).
- **Env**: `uv pip install fastapi "uvicorn[standard]" pydantic pillow torchvision matplotlib tqdm`.
