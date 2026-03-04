import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

MODEL_PATH = Path(__file__).resolve().parent.parent / "plant_disease_v2.keras"

model: tf.keras.Model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    yield


app = FastAPI(title="Plant Disease Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


class PredictResponse(BaseModel):
    cls: str
    confidence: float

    model_config = {"populate_by_name": True}

    def model_dump(self, **kwargs):
        d = super().model_dump(**kwargs)
        d["class"] = d.pop("cls")
        return d


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize(config.IMG_SIZE)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    arr = np.array(img, dtype=np.float32) / 255.0   # normalize to [0, 1]
    scores = model.predict(arr[np.newaxis], verbose=0)[0]
    idx = int(np.argmax(scores))

    return PredictResponse(cls=config.CLASS_NAMES[idx], confidence=round(float(scores[idx]), 4))
