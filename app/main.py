from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io, os, base64, math
import torch
from torchvision import transforms
from torchvision.utils import save_image

# ---------- Config for Classifier ----------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
TRANSFORM = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

from helper_lib.model import get_model

# Prefer mps -> cuda -> cpu
try:
    from helper_lib.trainer import select_device
except Exception:
    def select_device():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

app = FastAPI(title="Image Classifier + GAN API", version="1.1.0")

# ---------- Globals ----------
MODEL = None        # classifier
LABELS = None
GAN = None          # wrapper with generator/discriminator
GAN_Z_DIM = 100
DEVICE = select_device()

def _load_labels(path: str):
    with open(path, "r") as f:
        return [x.strip() for x in f if x.strip()]

@app.on_event("startup")
def _warmup():
    """Load classifier (required) and GAN generator (optional) on startup."""
    global MODEL, LABELS, GAN, DEVICE
    DEVICE = select_device()
    print(f"[startup] device = {DEVICE}")

    # ---- Load classifier ----
    try:
        LABELS = _load_labels("./artifacts/labels.txt")
        num_classes = len(LABELS)
        print(f"[startup] labels loaded: {num_classes} classes")

        MODEL = get_model("AssignmentCNN", num_classes=num_classes)
        MODEL.load_state_dict(torch.load("./artifacts/model.pt", map_location="cpu"))
        MODEL.eval()
        print("[startup] Classifier model & labels loaded.")
    except Exception as e:
        import traceback
        print("[startup] Classifier warmup failed:", e)
        traceback.print_exc()
        MODEL = None
        LABELS = None

    # ---- Load GAN generator (optional) ----
    try:
        if os.path.exists("./artifacts/gan_generator.pt"):
            GAN = get_model("gan", z_dim=GAN_Z_DIM)  # implemented in your helper_lib/model.py
            state = torch.load("./artifacts/gan_generator.pt", map_location=DEVICE)
            GAN.generator.load_state_dict(state)
            GAN.generator.to(DEVICE).eval()
            print("[startup] GAN generator loaded from ./artifacts/gan_generator.pt")
        else:
            GAN = get_model("gan", z_dim=GAN_Z_DIM)  # random init for API smoke test
            GAN.generator.to(DEVICE).eval()
            print("[startup] No gan_generator.pt found. Using randomly initialized generator.")
    except Exception as e:
        import traceback
        print("[startup] GAN warmup failed:", e)
        traceback.print_exc()
        GAN = None  # disable GAN endpoint if loading fails

# ---------- Classification Endpoint ----------
@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """Classify an uploaded image with your CIFAR10-like classifier."""
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    if MODEL is None or LABELS is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        x = TRANSFORM(img).unsqueeze(0)  # [1,3,64,64]
        with torch.no_grad():
            logits = MODEL(x)
            prob = torch.softmax(logits, dim=1)
            conf, idx = prob.max(dim=1)
            idx = idx.item()
            if idx < 0 or idx >= len(LABELS):
                raise IndexError(f"class index {idx} out of range for labels size {len(LABELS)}")
            return JSONResponse({"class": LABELS[idx], "confidence": float(conf.item())})
    except Exception as e:
        import traceback
        print("[/classify] inference error:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")

# ---------- GAN Generation ----------
class GanRequest(BaseModel):
    num_samples: int = 16        # best as perfect squares: 4, 9, 16, 25, ...
    z_dim: int | None = None     # override z-dim per request (optional)

@app.post("/generate_gan")
def generate_gan(req: GanRequest):
    """
    Generate an MNIST grid (Base64 PNG).
    - If ./artifacts/gan_generator.pt is present, it uses that trained generator.
      Otherwise it uses a randomly initialized generator (good for API testing).
    """
    if GAN is None:
        raise HTTPException(status_code=500, detail="GAN is not available")

    # If a different z_dim is requested, build a temporary generator
    z_dim = int(req.z_dim) if req.z_dim else GAN_Z_DIM
    if z_dim != getattr(GAN, "z_dim", GAN_Z_DIM):
        try:
            _tmp_gan = get_model("gan", z_dim=z_dim)
            _tmp_gan.generator.to(DEVICE).eval()
            generator = _tmp_gan.generator
        except Exception as e:
            print("[/generate_gan] fallback to existing GAN due to:", e)
            generator = GAN.generator
            z_dim = getattr(GAN, "z_dim", GAN_Z_DIM)
    else:
        generator = GAN.generator

    num = max(1, int(req.num_samples))
    nrow = int(math.sqrt(num))
    if nrow * nrow != num:
        nrow = max(1, int(round(math.sqrt(num))))

    with torch.no_grad():
        z = torch.randn(num, z_dim, device=DEVICE)
        imgs = generator(z).cpu()       # Tanh output in [-1, 1]
        imgs = (imgs + 1) / 2.0         # map back to [0, 1]

    # Encode as Base64 PNG
    buffer = io.BytesIO()
    save_image(imgs, buffer, nrow=nrow, format="PNG")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("utf-8")

    return {
        "num_samples": num,
        "z_dim": z_dim,
        "image_base64_png": b64
    }

@app.get("/")
def root():
    return {
        "status": "ok",
        "use": {
            "classify": "POST /classify (multipart/form-data: file=<image>)",
            "gan": "POST /generate_gan (json: {num_samples: 16, z_dim: 100})"
        },
        "docs": "/docs"
    }
