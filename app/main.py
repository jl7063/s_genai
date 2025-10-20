from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io, torch
from helper_lib.model import get_model
from helper_lib.evaluator import prepare_eval_transform

app = FastAPI(title="Image Classifier API", version="1.0.0")

MODEL = None
LABELS = None
TRANSFORM = prepare_eval_transform()

def _load_labels(path: str):
    with open(path, "r") as f:
        return [x.strip() for x in f if x.strip()]

@app.on_event("startup")
def _warmup():
    global MODEL, LABELS
    try:
        MODEL = get_model("CNN", num_classes=10)
        MODEL.load_state_dict(torch.load("./artifacts/model.pt", map_location="cpu"))
        MODEL.eval()
        LABELS = _load_labels("./artifacts/labels.txt")
        print("Model & labels loaded.")
    except Exception as e:
        print("Warmup failed:", e)

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    if MODEL is None or LABELS is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    x = TRANSFORM(img).unsqueeze(0)  # [1,3,32,32]
    with torch.no_grad():
        logits = MODEL(x)
        prob = torch.softmax(logits, dim=1)
        conf, idx = prob.max
@app.get("/")
def root():
    return {"status": "ok", "use": "POST /classify", "docs": "/docs"}
