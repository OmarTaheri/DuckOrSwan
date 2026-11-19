"""Duck vs Swan classifier - FastAPI web app (CPU inference)."""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

from io import BytesIO
from pathlib import Path

import keras
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from . import augment  # noqa: F401  registers the GPUAugment layer (identity at inference)

BASE = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE.parent / "model" / "duck_swan_model.keras"))
IMG_SIZE = (224, 224)  # replaced by the model's real input size at startup
MAX_BYTES = 10 * 1024 * 1024
# image_dataset_from_directory sorts class folders alphabetically: 0 = duck, 1 = swan
CLASS_NAMES = ["Duck", "Swan"]

app = FastAPI(title="Duck vs Swan Classifier")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
model = None


@app.on_event("startup")
def load_model():
    global model, IMG_SIZE
    model = keras.models.load_model(MODEL_PATH)
    IMG_SIZE = tuple(model.input_shape[1:3])
    model.predict(np.zeros((1, *IMG_SIZE, 3), dtype="float32"), verbose=0)  # warm-up


def preprocess(raw: bytes) -> np.ndarray:
    # Same steps as the lecture: RGB -> resize to the model input size -> /255 -> expand_dims
    img = Image.open(BytesIO(raw)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "Image too large (max 10 MB)")
    try:
        x = preprocess(raw)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "File is not a valid image")
    p_swan = float(model.predict(x, verbose=0)[0][0])
    if p_swan > 0.5:
        label, confidence = CLASS_NAMES[1], p_swan
    else:
        label, confidence = CLASS_NAMES[0], 1 - p_swan
    return {"label": label, "confidence": round(confidence, 4), "p_swan": round(p_swan, 4)}
