from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io, torch
from torchvision import transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
TRANSFORM = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

from helper_lib.model import get_model

app = FastAPI(title="Image Classifier API", version="1.0.0")

MODEL = None
LABELS = None

def _load_labels(path: str):
    with open(path, "r") as f:
        return [x.strip() for x in f if x.strip()]

@app.on_event("startup")
def _warmup():
    global MODEL, LABELS
    try:
        LABELS = _load_labels("./artifacts/labels.txt")
        num_classes = len(LABELS)
        print(f"[startup] labels loaded: {num_classes} classes")

        MODEL = get_model("AssignmentCNN", num_classes=num_classes)
        MODEL.load_state_dict(torch.load("./artifacts/model.pt", map_location="cpu"))
        MODEL.eval()
        print("Model & labels loaded.")
    except Exception as e:
        import traceback
        print("Warmup failed:", e)
        traceback.print_exc()
        MODEL = None
        LABELS = None

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
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

@app.get("/")
def root():
    return {"status": "ok", "use": "POST /classify", "docs": "/docs"}
