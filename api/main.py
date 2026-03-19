import io
import os
import sys
import json
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

import google.genai as genai
from google.genai import types as genai_types
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

# Load api/.env (local dev). In production (Railway) env vars are set via the dashboard.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

# gdown handles Google Drive large-file downloads (Drive shows virus-scan page for >100MB)
try:
    import gdown
except ImportError:
    gdown = None
from fastapi import Depends, FastAPI, File, HTTPException, Request, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# Local dev: model at project root, Prev Models/, or api/. Docker: often gets Git LFS pointer (invalid).
# Set MODEL_URL to download the real model at startup when the bundled file is missing or invalid.
_base = Path(__file__).resolve().parent
_root = _base.parent
_candidates = [
    _root / "plant_disease_v2.keras",
    _root / "trained_plant_disease_model.h5",
    _root / "Prev Models" / "trained_plant_disease_model.h5",
    _base / "plant_disease_v2.keras",
    _base / "trained_plant_disease_model.h5",
]
MODEL_PATH = next((p for p in _candidates if p.exists()), _base / "trained_plant_disease_model.h5")

# Min size for a valid HDF5 model (~193MB). Git LFS pointer is ~200 bytes.
MIN_MODEL_BYTES = 50_000_000

model: tf.keras.Model | None = None
tflite_model: tf.lite.Interpreter | None = None

class _DTypePolicyShim:
    """
    Lets Keras deserialize legacy DTypePolicy configs. Every layer (Conv2D, Dense, etc.)
    has dtype: {'class_name': 'DTypePolicy', 'config': {'name': 'float32'}} in its config;
    registering this in custom_object_scope fixes all of them at once.
    """

    @classmethod
    def from_config(cls, config):
        if not isinstance(config, dict):
            return tf.keras.mixed_precision.Policy("float32")
        inner = config.get("config", config)
        name = inner.get("name", "float32") if isinstance(inner, dict) else "float32"
        return tf.keras.mixed_precision.Policy(name)


def _clean_legacy_config(config: dict) -> dict:
    """
    Normalize legacy Keras layer configs:
    - Convert DTypePolicy dicts to plain dtype strings (e.g. 'float32')
    - Drop obsolete arguments like 'data_format' that newer layers reject.
    """
    cfg = dict(config)
    dtype_cfg = cfg.get("dtype")
    if isinstance(dtype_cfg, dict) and dtype_cfg.get("class_name") == "DTypePolicy":
        cfg["dtype"] = dtype_cfg.get("config", {}).get("name", "float32")
    # These older preprocessing layers used to accept 'data_format' in config;
    # newer versions often do not.
    cfg.pop("data_format", None)
    return cfg


class _LegacyInputLayer(tf.keras.layers.InputLayer):
    """
    Compatibility shim for models saved with old tf.keras that used 'batch_shape'.
    We convert it to batch_input_shape so the first Conv2D gets a defined channel dim.
    """

    @classmethod
    def from_config(cls, config):
        cfg = dict(config)
        batch_shape = cfg.pop("batch_shape", None)
        if batch_shape is not None:
            cfg["batch_input_shape"] = batch_shape
        cfg = _clean_legacy_config(cfg)
        return super().from_config(cfg)


class _LegacyRescaling(tf.keras.layers.Rescaling):
    """
    Compatibility shim for models whose Rescaling layer was saved with a 'dtype'
    config pointing to the now-unknown 'DTypePolicy' object. We map that to a
    plain dtype string (e.g. 'float32') so modern Keras deserialization works.
    """

    @classmethod
    def from_config(cls, config):
        cfg = _clean_legacy_config(config)
        return super().from_config(cfg)


class _LegacyRandomFlip(tf.keras.layers.RandomFlip):
    """
    Compatibility shim for RandomFlip layers saved with legacy dtype policies and
    an obsolete 'data_format' argument.
    """

    @classmethod
    def from_config(cls, config):
        cfg = _clean_legacy_config(config)
        return super().from_config(cfg)


class _LegacyRandomRotation(tf.keras.layers.RandomRotation):
    """
    Compatibility shim for RandomRotation layers saved with legacy dtype policies and
    obsolete 'data_format' argument.
    """

    @classmethod
    def from_config(cls, config):
        cfg = _clean_legacy_config(config)
        return super().from_config(cfg)


class _LegacyRandomZoom(tf.keras.layers.RandomZoom):
    """
    Compatibility shim for RandomZoom layers with legacy dtype/data_format config.
    """

    @classmethod
    def from_config(cls, config):
        cfg = _clean_legacy_config(config)
        return super().from_config(cfg)


def _load_legacy_model(path: Path) -> tf.keras.Model:
    custom_objects = {
        "DTypePolicy": _DTypePolicyShim,
        "InputLayer": _LegacyInputLayer,
        "Rescaling": _LegacyRescaling,
        "RandomFlip": _LegacyRandomFlip,
        "RandomRotation": _LegacyRandomRotation,
        "RandomZoom": _LegacyRandomZoom,
    }
    with tf.keras.utils.custom_object_scope(custom_objects):
        # compile=False skips optimizer deserialization entirely — safe for inference-only use
        return tf.keras.models.load_model(path, compile=False)


# Fallback model URL when bundled file is missing or a Git LFS pointer (~200 bytes).
_DEFAULT_MODEL_URL = "https://drive.google.com/uc?id=1N1eBI4xpBEOGlWcdIEqSak8nF6n4yDJH"


def _get_model_path() -> Path:
    """Return path to a valid model file, downloading from MODEL_URL if needed."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size >= MIN_MODEL_BYTES:
        return MODEL_PATH
    url = os.getenv("MODEL_URL") or _DEFAULT_MODEL_URL
    if url:
        dest = _base / "trained_plant_disease_model.h5"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if "drive.google.com" in url and gdown:
            gdown.download(url, str(dest), quiet=False, fuzzy=True)
        else:
            urllib.request.urlretrieve(url, dest)
        return dest
    raise RuntimeError(
        f"Model not found or invalid (Git LFS pointer?). "
        "Set MODEL_URL to a direct download URL, or ensure the real .h5 file is in the image."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tflite_model
    try:
        path = _get_model_path()
        model = _load_legacy_model(path)
        # Log the first layer so the normalization contract is visible at every startup.
        first_layer = model.layers[0]
        scale = getattr(first_layer, "scale", "N/A")
        print(
            f"[startup] model loaded from {path.name!r} | "
            f"first_layer={type(first_layer).__name__} scale={scale} | "
            f"expected_input_range=[0, 255]"
        )
    except Exception as e:
        print(f"[startup] Failed to load main model: {e}. Falling back to offline TFLite model...")
        tflite_path = _base.parent / "plant_disease_model.tflite"
        if not tflite_path.exists():
            tflite_path = _base / "plant_disease_model.tflite"
        if tflite_path.exists():
            tflite_model = tf.lite.Interpreter(model_path=str(tflite_path))
            tflite_model.allocate_tensors()
            print(f"[startup] Loaded offline TFLite model from {tflite_path.name!r}.")
        else:
            print("[startup] Warning: No models available.")
    yield


app = FastAPI(title="Plant Disease Detection API", lifespan=lifespan)

# ── Rate limiting (slowapi) ───────────────────────────────────────────────────
# Default: 10 predict requests per minute per IP.
# Override with RATE_LIMIT env var, e.g. "20/minute" or "100/hour".
_rate_limit = os.getenv("RATE_LIMIT", "10/minute")
limiter = Limiter(key_func=get_remote_address, default_limits=[_rate_limit])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

# ── API key auth ──────────────────────────────────────────────────────────────
# Set API_KEY env var on Railway. If unset, the endpoint is open (dev mode).
_API_KEY = os.getenv("API_KEY", "").strip() or None
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def _require_api_key(key: str | None = Security(_api_key_header)) -> None:
    if _API_KEY and key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

# ── CORS ─────────────────────────────────────────────────────────────────────
# Set CORS_ORIGINS env var (comma-separated). Defaults to * (open).
_cors_env = os.getenv("CORS_ORIGINS", "*").strip()
_origins_list = ["*"] if _cors_env == "*" else [o.strip() for o in _cors_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Input validation constants ────────────────────────────────────────────────
_MAX_FILE_BYTES = 10 * 1024 * 1024   # 10 MB
_MAX_IMAGE_DIM  = 4096               # max width or height in pixels

# Magic-byte signatures for accepted image types
_MAGIC: dict[bytes, str] = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"RIFF": "image/webp",  # WebP starts with RIFF
    b"GIF8": "image/gif",
}

def _validate_image_bytes(raw: bytes) -> str | None:
    """Return error message if bytes don't look like a supported image, else None."""
    if len(raw) > _MAX_FILE_BYTES:
        return f"File too large. Maximum size is {_MAX_FILE_BYTES // 1_048_576} MB."
    for magic, _ in _MAGIC.items():
        if raw[:len(magic)] == magic:
            return None
    return "Invalid file format. Please upload a PNG or JPG image."


class PredictionItem(BaseModel):
    label: str
    confidence: float
    treatment: list[str] = Field(default_factory=list)
    prevention: list[str] = Field(default_factory=list)

class PredictResponse(BaseModel):
    predictions: list[PredictionItem]


class ChatRequest(BaseModel):
    predictions: list[PredictionItem]
    message: str


class ChatResponse(BaseModel):
    reply: str


# Fraction of pixels that must look plant-like before we even run the model.
_MIN_LEAF_PIXEL_RATIO = 0.08
# Model confidence below this is treated as "not a recognisable leaf".
_MIN_CONFIDENCE = 0.20

_treatments_file = Path(__file__).resolve().parent / "treatments.json"
_treatments_db = {}
if _treatments_file.exists():
    with open(_treatments_file, "r") as f:
        _treatments_db = json.load(f)

_analytics_file = Path(__file__).resolve().parent / "analytics.json"
_analytics_db = {}
if _analytics_file.exists():
    try:
        with open(_analytics_file, "r") as f:
            _analytics_db = json.load(f)
    except Exception:
        _analytics_db = {}


def _is_likely_leaf(img: Image.Image) -> bool:
    """
    Fast colour heuristic: checks whether enough pixels fall in ranges
    typical of plant leaves (green for healthy; yellow/brown for diseased).
    No extra dependencies — uses only numpy + PIL which are already loaded.
    """
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # Healthy leaves: green channel clearly dominant
    green = (g > 60) & (g > r * 1.05) & (g > b * 1.05)

    # Diseased / dry leaves: yellow-green or brown tones
    yellow_brown = (r > 80) & (g > 60) & (b < 130) & (r > b * 1.15) & (g > b * 1.05)

    ratio = float(np.sum(green | yellow_brown)) / (arr.shape[0] * arr.shape[1])
    return ratio >= _MIN_LEAF_PIXEL_RATIO


@app.get("/health")
def health():
    """Health check for Railway / load balancers."""
    return {"status": "ok"}


_SYSTEM_PROMPT = (
    "You are a practical farming assistant. "
    "The plant disease model evaluated the farmer's leaf and gave these predictions:\n"
    "{predictions_context}\n\n"
    "Explain the most likely disease (the main one). If other predictions are close in confidence, briefly mention them as other possibilities. "
    "Answer in plain, simple language — no scientific jargon unless truly needed. "
    "Do NOT use markdown bold formatting (like **). "
    "Keep every reply under 120 words. "
    "Structure your answer with these three short bullet sections:\n"
    "- Severity: Is this serious? Will it spread fast or destroy the crop?\n"
    "- Treatment: What should the farmer do RIGHT NOW to stop it?\n"
    "- Prevention: How to stop it coming back next season?\n"
    "Be direct and practical. Skip lengthy explanations."
)


def _call_gemini(api_key: str, message: str, predictions_context: str) -> tuple[str | None, str | None]:
    """Try Gemini 2.5 Flash, fallback to 2.0. Returns (reply, error_msg)."""
    client = genai.Client(api_key=api_key)
    config = genai_types.GenerateContentConfig(
        system_instruction=_SYSTEM_PROMPT.format(predictions_context=predictions_context),
    )
    for model_id in ("gemini-2.5-flash", "gemini-2.0-flash"):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=message,
                config=config,
            )
            return (response.text, None)
        except Exception as exc:
            err = str(exc)
            print(f"[chat] Gemini ({model_id}) error: {err}")
    return (None, err)


def _call_openai(api_key: str, message: str, predictions_context: str) -> tuple[str | None, str | None]:
    """Try OpenAI (gpt-4.1-nano cheapest, fallback gpt-4o-mini). Returns (reply, error_msg)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT.format(predictions_context=predictions_context)},
        {"role": "user", "content": message},
    ]
    for model_id in ("gpt-4.1-nano", "gpt-4o-mini"):
        try:
            response = client.chat.completions.create(model=model_id, messages=messages)
            return (response.choices[0].message.content, None)
        except Exception as exc:
            err = str(exc)
            print(f"[chat] OpenAI ({model_id}) error: {err}")
    return (None, err)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    """Ask Gemini 2.5 Flash about the detected plant disease. Falls back to OpenAI if Gemini fails."""
    gemini_key_header = request.headers.get("x-gemini-key", "").strip()
    openai_key_header = request.headers.get("x-openai-key", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip() or gemini_key_header
    openai_key = os.getenv("OPENAI_API_KEY", "").strip() or openai_key_header

    if not gemini_key and not openai_key:
        raise HTTPException(
            status_code=503,
            detail="Neither GEMINI_API_KEY nor OPENAI_API_KEY is configured on the server.",
        )

    predictions_context = "\n".join(
        f"- {p.label} ({p.confidence * 100:.1f}%)" for p in body.predictions
    ) or "No predictions available."

    reply: str | None = None
    errors: list[str] = []

    if gemini_key:
        reply, err = _call_gemini(gemini_key, body.message, predictions_context)
        if err:
            errors.append(f"Gemini: {err[:100]}")

    if reply is None and openai_key:
        reply, err = _call_openai(openai_key, body.message, predictions_context)
        if err:
            errors.append(f"OpenAI: {err[:100]}")

    if reply is None:
        hint = " Ensure GEMINI_API_KEY and/or OPENAI_API_KEY are set correctly on the server."
        if errors:
            hint = f" {errors[0]}" if len(errors) == 1 else f" Last errors: {'; '.join(errors[-2:])}"
        raise HTTPException(
            status_code=502,
            detail=f"Both Gemini and OpenAI chat services failed.{hint}",
        )

    return ChatResponse(reply=reply)


@app.post(
    "/predict",
    response_model=PredictResponse,
    dependencies=[Depends(_require_api_key)],
)
@limiter.limit(_rate_limit)
async def predict(request: Request, file: UploadFile = File(...)):  # noqa: ARG001
    # ── Filename / content-type sanity check ─────────────────────────────────
    filename = (file.filename or "").lower()
    allowed_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    if not any(filename.endswith(ext) for ext in allowed_exts):
        if not (file.content_type or "").startswith("image/"):
            return JSONResponse(status_code=400, content={"error": "Invalid file format. Please upload a PNG or JPG image."})

    # ── Read and hard-limit file size ─────────────────────────────────────────
    raw = await file.read()
    print(f"[predict] received: filename={file.filename!r} content_type={file.content_type!r} bytes={len(raw)}")
    
    val_err = _validate_image_bytes(raw)
    if val_err:
        return JSONResponse(status_code=400, content={"error": val_err})

    # ── Decode image ──────────────────────────────────────────────────────────
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Corrupted or invalid image file. Please upload a valid PNG or JPG photo."})

    # ── Dimension guard ───────────────────────────────────────────────────────
    w, h = img.size
    if w > _MAX_IMAGE_DIM or h > _MAX_IMAGE_DIM:
        return JSONResponse(
            status_code=400,
            content={"error": f"Image dimensions too large ({w}×{h}). Maximum is {_MAX_IMAGE_DIM}×{_MAX_IMAGE_DIM} pixels."},
        )

    # ── Image Quality Validation ──────────────────────────────────────────────
    if w < 128 or h < 128:
        return JSONResponse(status_code=400, content={"error": "Image quality too low. Please upload a clear close-up leaf photo."})
    
    img_gray = np.array(img.convert("L"))
    if np.mean(img_gray) < 20: 
        return JSONResponse(status_code=400, content={"error": "Image quality too low. Please upload a clear close-up leaf photo."})
    if np.var(img_gray) < 10: 
        return JSONResponse(status_code=400, content={"error": "Image quality too low. Please upload a clear close-up leaf photo."})

    # ── Leaf colour heuristic ─────────────────────────────────────────────────
    if not _is_likely_leaf(img):
        raise HTTPException(
            status_code=422,
            detail="No plant leaf detected. Please upload a clear photo of a leaf.",
        )

    # ── Model inference ───────────────────────────────────────────────────────
    # PIL resize takes (width, height); config.IMG_SIZE is (H, W) = (128, 128).
    img_h, img_w = config.IMG_SIZE
    img_resized = img.resize((img_w, img_h), resample=Image.BILINEAR)

    # Input range: [0, 255] float32.
    # The model's first layer is Rescaling(1/255), so normalization happens inside
    # the model — do NOT divide here or predictions will be wrong (double-normalized).
    arr = np.array(img_resized, dtype=np.float32)

    print(
        f"[predict] file={file.filename!r} "
        f"original_size={img.size} resized=({img_w},{img_h}) "
        f"arr_shape={arr.shape} arr_min={arr.min():.1f} arr_max={arr.max():.1f}"
    )

    if model is not None:
        scores = model.predict(arr[np.newaxis], verbose=0)[0]
    elif tflite_model is not None:
        input_details = tflite_model.get_input_details()
        output_details = tflite_model.get_output_details()
        tflite_model.set_tensor(input_details[0]['index'], arr[np.newaxis])
        tflite_model.invoke()
        scores = tflite_model.get_tensor(output_details[0]['index'])[0]
    else:
        raise HTTPException(status_code=503, detail="Prediction models are currently unavailable.")

    # Get top-3 predictions
    top3_idx = np.argsort(scores)[::-1][:3]
    top3 = [(config.CLASS_NAMES[i], round(float(scores[i]), 4)) for i in top3_idx]
    
    idx = top3_idx[0]
    confidence = top3[0][1]
    
    print(f"[predict] chosen_idx={idx} class={config.CLASS_NAMES[idx]!r} confidence={confidence} top3={top3}")

    if confidence < _MIN_CONFIDENCE:
        raise HTTPException(
            status_code=422,
            detail="Image not recognised as a plant leaf. Please upload a clear, well-lit leaf photo.",
        )

    predictions = []
    for label, conf in top3:
        info = _treatments_db.get(label, {})
        predictions.append(
            PredictionItem(
                label=label, 
                confidence=conf,
                treatment=info.get("treatment", []),
                prevention=info.get("prevention", [])
            )
        )
        
    _analytics_db[config.CLASS_NAMES[idx]] = _analytics_db.get(config.CLASS_NAMES[idx], 0) + 1
    with open(_analytics_file, "w") as f:
        json.dump(_analytics_db, f)

    return PredictResponse(predictions=predictions)

@app.get("/analytics")
def get_analytics():
    return {"disease_counts": _analytics_db}
