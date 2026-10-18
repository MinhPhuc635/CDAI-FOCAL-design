"""FOCAL MOD preprocessing and inference helpers shared by training and Streamlit."""
from __future__ import annotations

import json
import os
import sys
import wave
from pathlib import Path

import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch
from scipy.signal import resample_poly

torch.set_num_threads(max(1, int(os.getenv("FOCAL_TORCH_THREADS", "1"))))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # PyTorch only allows this setting before its inter-op pool is initialized.
    pass

ROOT = Path(__file__).resolve().parent
FOCAL_SRC = ROOT / "focal-main" / "src"
if str(FOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(FOCAL_SRC))

MODEL_CONFIG = FOCAL_SRC / "data" / "MOD.yaml"
MODEL_PATH = Path(os.getenv("FOCAL_MODEL_PATH", str(ROOT / "models" / "vehicle_focal.pt")))
LABELS_PATH = Path(os.getenv("FOCAL_LABELS_PATH", str(ROOT / "models" / "labels.json")))
METRICS_PATH = ROOT / "models" / "model_metrics.json"

INPUT_RATE = 16_000
AUDIO_RATE = 8_000
SEISMIC_RATE = 100
WINDOW_SECONDS = 2
INTERVAL_SECONDS = 0.2
NUM_INTERVALS = 10


def read_audio_wav(source) -> np.ndarray:
    """Read mono/stereo PCM WAV as float values on the original integer scale."""
    if hasattr(source, "seek"):
        source.seek(0)
    with wave.open(source, "rb") as wav:
        channels, sample_width, sample_rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
        if sample_rate != INPUT_RATE:
            raise ValueError(f"Audio phải có tần số lấy mẫu 16 kHz; file hiện tại là {sample_rate} Hz.")
        raw = wav.readframes(wav.getnframes())
    if sample_width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32)
    elif sample_width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0)
    else:
        raise ValueError(f"WAV dùng định dạng PCM {sample_width * 8}-bit chưa được hỗ trợ.")
    if channels > 1:
        audio = audio.reshape(-1, channels)[:, 0]
    return resample_poly(audio, up=1, down=2).astype(np.float32, copy=False)


def read_audio_csv(source) -> np.ndarray:
    """Read the first acoustic channel from upstream MOD aud*.csv recordings."""
    if hasattr(source, "seek"):
        source.seek(0)
    # MOD acoustic CSV is comma delimited; the upstream extractor consumes column 0.
    try:
        audio = np.loadtxt(source, delimiter=",", usecols=(0,), dtype=np.float32)
    except ValueError:
        if hasattr(source, "seek"):
            source.seek(0)
        audio = np.loadtxt(source, usecols=(0,), dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Không đọc được tín hiệu âm thanh từ CSV.")
    return resample_poly(audio, up=1, down=2).astype(np.float32, copy=False)


def read_seismic_csv(source) -> np.ndarray:
    """Read the vertical seismic component used by FOCAL MOD (ehz.csv)."""
    if hasattr(source, "seek"):
        source.seek(0)
    try:
        values = np.loadtxt(source, usecols=(0,), dtype=np.float32)
    except ValueError:
        if hasattr(source, "seek"):
            source.seek(0)
        values = np.loadtxt(source, delimiter=",", usecols=(0,), dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Không đọc được tín hiệu địa chấn từ ehz.csv.")
    return values


def make_time_windows(audio_8k: np.ndarray, seismic_100hz: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FOCAL time inputs [N, 1, 10, 1600] and [N, 1, 10, 20]."""
    length = min(audio_8k.size // (AUDIO_RATE * WINDOW_SECONDS), seismic_100hz.size // (SEISMIC_RATE * WINDOW_SECONDS))
    if length < 1:
        raise ValueError("Hai tín hiệu cần có ít nhất 2 giây dữ liệu đồng bộ.")
    audio = audio_8k[: length * AUDIO_RATE * WINDOW_SECONDS]
    seismic = seismic_100hz[: length * SEISMIC_RATE * WINDOW_SECONDS]
    audio = audio.reshape(length, NUM_INTERVALS, int(INTERVAL_SECONDS * AUDIO_RATE))[:, None, :, :]
    seismic = seismic.reshape(length, NUM_INTERVALS, int(INTERVAL_SECONDS * SEISMIC_RATE))[:, None, :, :]
    # The arrays are already views into contiguous, decoded buffers. Avoid a
    # second full-size copy; inference processes these windows in small batches.
    return torch.from_numpy(audio), torch.from_numpy(seismic)


def time_to_frequency(audio: torch.Tensor, seismic: torch.Tensor) -> dict:
    """Apply the same per-interval complex FFT transform used by FOCAL's Augmenter."""
    def convert(tensor: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fft(tensor.float(), dim=-1)
        spectrum = torch.view_as_real(spectrum).permute(0, 1, 4, 2, 3)
        batch, channels, real_imag, intervals, bins = spectrum.shape
        return spectrum.reshape(batch, channels * real_imag, intervals, bins)

    return {"shake": {"audio": convert(audio), "seismic": convert(seismic)}}


def load_focal_model(device: str = "cpu"):
    """Build the upstream FOCAL SW-Transformer classifier and load its fine-tuned weights."""
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError("Checkpoint FOCAL hoặc labels.json chưa được tạo.")
    import yaml
    from types import SimpleNamespace
    from models.DeepSense import DeepSense

    config = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    args = SimpleNamespace(
        dataset="MOD", model="DeepSense", task="vehicle_classification",
        train_mode="contrastive", stage="finetune", device=torch.device(device),
        dataset_config=config,
    )
    model = DeepSense(args).to(device)
    try:
        state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except TypeError:  # Torch versions before weights_only was introduced.
        state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, labels


@torch.inference_mode()
def predict_windows(model, audio: torch.Tensor, seismic: torch.Tensor, batch_size: int = 4) -> np.ndarray:
    """Average per-window softmax probabilities across a recording."""
    probability_sum = None
    total_windows = 0
    for start in range(0, len(audio), batch_size):
        end = min(start + batch_size, len(audio))
        freq = time_to_frequency(audio[start:end], seismic[start:end])
        logits = model(freq)
        batch_probabilities = torch.softmax(logits, dim=1)
        batch_sum = batch_probabilities.sum(dim=0).cpu()
        probability_sum = batch_sum if probability_sum is None else probability_sum + batch_sum
        total_windows += end - start
        del freq, logits, batch_probabilities, batch_sum
    if probability_sum is None:
        raise ValueError("Không có cửa sổ tín hiệu hợp lệ để dự đoán.")
    return (probability_sum / total_windows).numpy()

