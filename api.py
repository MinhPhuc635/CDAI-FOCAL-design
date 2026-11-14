"""REST API for FOCAL vehicle classification.

Run locally with: uvicorn api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import hmac
import io
import json
import logging
import os
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import focal_pipeline as focal


logger = logging.getLogger("focal.api")
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 24 * 1024 * 1024
METRICS_PATH = Path(__file__).parent / "models" / "model_metrics.json"

app = FastAPI(
    title="FOCAL Vehicle Classifier API",
    description="Classify a synchronized audio and vertical seismic recording.",
    version="1.0.0",
)
# Same-origin requests work without CORS. Set FOCAL_CORS_ORIGINS to a comma-
# separated allowlist only when a separate browser frontend needs API access.
cors_origins = [origin.strip() for origin in os.getenv("FOCAL_CORS_ORIGINS", "").split(",") if origin.strip()]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )


def _check_api_key(x_api_key: str | None) -> None:
    """If configured, require the key without ever returning its value."""
    expected = os.getenv("FOCAL_API_KEY")
    if expected and (not x_api_key or not hmac.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


@lru_cache(maxsize=1)
def _model_and_labels():
    # Do not trigger training from an inference request. The checkpoint must
    # be built and packaged/deployed by the admin beforehand.
    if not focal.MODEL_PATH.is_file() or not focal.LABELS_PATH.is_file():
        raise FileNotFoundError("FOCAL checkpoint or labels are not installed.")
    return focal.load_focal_model("cpu")


async def _read_limited(upload: UploadFile, kind: str) -> bytes:
    name = Path(upload.filename or "").name
    suffix = Path(name).suffix.lower()
    allowed = {".wav", ".csv"} if kind == "audio" else {".csv"}
    if suffix not in allowed:
        expected = ".wav or .csv" if kind == "audio" else ".csv"
        raise HTTPException(status_code=415, detail=f"{kind} file must be {expected}.")
    content = await upload.read(MAX_FILE_BYTES + 1)
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="Each uploaded file is limited to 16 MiB.")
    if not content:
        raise HTTPException(status_code=422, detail=f"The {kind} file is empty.")
    return content


@app.get("/api/v1/health")
def health():
    checkpoint_ready = focal.MODEL_PATH.is_file() and focal.LABELS_PATH.is_file()
    return {
        "status": "ready" if checkpoint_ready else "needs_checkpoint",
        "model": "FOCAL",
        "checkpoint_ready": checkpoint_ready,
    }


@app.post("/api/v1/predict")
async def predict(
    audio: UploadFile = File(..., description="Synchronized .wav or MOD acoustic CSV (aud*.csv)."),
    seismic: UploadFile = File(..., description="Matching vertical EHZ seismic CSV."),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Return a vehicle class prediction for one synchronized recording pair."""
    _check_api_key(x_api_key)
    audio_bytes = await _read_limited(audio, "audio")
    seismic_bytes = await _read_limited(seismic, "seismic")
    if len(audio_bytes) + len(seismic_bytes) > MAX_TOTAL_BYTES:
        raise HTTPException(status_code=413, detail="The combined upload is limited to 24 MiB.")

    try:
        if Path(audio.filename or "").suffix.lower() == ".wav":
            audio_signal = focal.read_audio_wav(io.BytesIO(audio_bytes))
        else:
            audio_signal = focal.read_audio_csv(io.BytesIO(audio_bytes))
        seismic_signal = focal.read_seismic_csv(io.BytesIO(seismic_bytes))
        audio_windows, seismic_windows = focal.make_time_windows(audio_signal, seismic_signal)
    except (ValueError, OSError, EOFError, wave.Error) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        model, labels = _model_and_labels()
        probabilities = focal.predict_windows(model, audio_windows, seismic_windows, batch_size=4)
        if len(labels) != len(probabilities) or not np.isfinite(probabilities).all():
            raise RuntimeError("Model output does not match the configured labels.")
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.is_file() else {}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="FOCAL checkpoint is unavailable; ask the admin to install it.") from exc
    except Exception as exc:
        logger.exception("FOCAL inference failed")
        raise HTTPException(status_code=500, detail="FOCAL could not complete this prediction.") from exc

    order = np.argsort(probabilities)[::-1]
    top_index = int(order[0])
    confidence = float(probabilities[top_index])
    margin = confidence - float(probabilities[int(order[1])]) if len(order) > 1 else confidence
    unknown_reason = focal.detect_out_of_distribution_signal(audio_signal, seismic_signal)
    unknown = (
        unknown_reason is not None
        or confidence < float(metrics.get("ood_min_confidence", 0.30))
        or margin < float(metrics.get("ood_min_margin", 0.08))
    )
    return {
        "predicted_class": None if unknown else labels[top_index],
        "nearest_class": labels[top_index],
        "unknown": unknown,
        "unknown_reason": unknown_reason if unknown_reason else ("Weak signal or ambiguous classes." if unknown else None),
        "confidence": confidence,
        "confidence_percent": round(confidence * 100, 2),
        "classes": [
            {"label": labels[int(i)], "probability": float(probabilities[int(i)])}
            for i in order[: min(5, len(order))]
        ],
        "windows": int(len(audio_windows)),
        "test_accuracy_percent": metrics.get("test_accuracy_percent"),
        "confidence_note": "Confidence is the model probability, not a guarantee of correctness.",
    }
