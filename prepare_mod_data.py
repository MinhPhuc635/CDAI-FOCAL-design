"""Create train/validation/test tensors for the MOD_vehicle recordings."""
from __future__ import annotations

import json
import os
import random
import re
import wave
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.signal import resample_poly

from focal_pipeline import AUDIO_RATE, INPUT_RATE, INTERVAL_SECONDS, MODEL_CONFIG, NUM_INTERVALS, ROOT, WINDOW_SECONDS


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        channels, width, sample_rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
        if sample_rate != INPUT_RATE:
            raise ValueError(f"{path}: expected 16 kHz WAV, got {sample_rate} Hz")
        raw = wav.readframes(wav.getnframes())
    dtype = {1: np.uint8, 2: "<i2", 4: "<i4"}.get(width)
    if dtype is None:
        raise ValueError(f"Unsupported PCM width: {width} bytes ({path})")
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if width == 1:
        audio -= 128
    if channels > 1:
        audio = audio.reshape(-1, channels)[:, 0]
    return resample_poly(audio, 1, 2).astype(np.float32, copy=False)


def load_seismic(path: Path) -> np.ndarray:
    return np.loadtxt(path, usecols=(0,), dtype=np.float32)


def split_runs(class_names: list[str], recordings_by_class: dict[str, list[Path]], seed: int) -> dict[str, dict[str, list[Path]]]:
    rng = random.Random(seed)
    split = {"train": {}, "val": {}, "test": {}}
    for class_name in class_names:
        recordings = list(recordings_by_class[class_name])
        rng.shuffle(recordings)
        if len(recordings) >= 3:
            test_count = max(1, round(len(recordings) * 0.2))
            val_count = max(1, round(len(recordings) * 0.15))
            train_count = len(recordings) - test_count - val_count
            split["train"][class_name] = recordings[:train_count]
            split["val"][class_name] = recordings[train_count:train_count + val_count]
            split["test"][class_name] = recordings[train_count + val_count:]
        elif len(recordings) == 2:
            # Keep one complete recording for an independent test of sparse classes.
            split["train"][class_name] = recordings[:1]
            split["val"][class_name] = []
            split["test"][class_name] = recordings[1:]
        else:
            # A single recording cannot provide an independent run-level split.
            # Split it into contiguous time blocks to reduce adjacent-window leakage.
            split["train"][class_name] = recordings
            split["val"][class_name] = []
            split["test"][class_name] = []
    return split


def extract_recording(class_index: int, class_name: str, recording: Path, output_dir: Path) -> tuple[list[Path], int]:
    wavs = sorted(recording.glob("*.wav"))
    seis_path = recording / "ehz.csv"
    if not wavs or not seis_path.exists():
        return [], 0
    audio = load_wav(wavs[0])
    seismic = load_seismic(seis_path)
    sample_count = min(audio.size, (seismic.size // (100 * WINDOW_SECONDS)) * (AUDIO_RATE * WINDOW_SECONDS))
    sample_count = (sample_count // (AUDIO_RATE * WINDOW_SECONDS)) * AUDIO_RATE * WINDOW_SECONDS
    window_count = sample_count // (AUDIO_RATE * WINDOW_SECONDS)
    seismic_count = window_count * 100 * WINDOW_SECONDS
    if window_count == 0:
        return [], 0

    audio = audio[:sample_count].reshape(window_count, NUM_INTERVALS, AUDIO_RATE * WINDOW_SECONDS // NUM_INTERVALS)
    seismic = seismic[:seismic_count].reshape(window_count, NUM_INTERVALS, 100 * WINDOW_SECONDS // NUM_INTERVALS)
    outputs = []
    run_name = f"{recording.parent.name}__{recording.name}".replace("_", "-")
    for idx in range(window_count):
        # For single-run classes, contiguous windows are split by time below.
        sample_name = f"{class_name}__{run_name}__{idx:06d}.pt"
        path = output_dir / sample_name
        sample = {
            "label": torch.tensor(class_index).long(),
            "flag": {"shake": {"audio": True, "seismic": True}},
            "data": {
                "shake": {
                    "audio": torch.from_numpy(audio[idx:idx + 1].copy()),
                    "seismic": torch.from_numpy(seismic[idx:idx + 1].copy()),
                }
            },
        }
        if not path.exists():
            torch.save(sample, path)
        outputs.append(path)
    return outputs, window_count


def main() -> None:
    data_dir = Path(os.getenv("MOD_VEHICLE_DATA_DIR", r"F:\MOD_vehicle\raw_data"))
    if not data_dir.is_dir():
        raise FileNotFoundError(f"MOD_vehicle raw_data folder not found: {data_dir}")
    class_dirs: dict[str, list[Path]] = {}
    for folder in sorted((p for p in data_dir.iterdir() if p.is_dir()), key=lambda p: p.name.casefold()):
        label = re.sub(r"\d+$", "", folder.name).casefold()
        class_dirs.setdefault(label, []).append(folder)
    class_names = sorted(class_dirs)
    if len(class_names) < 2:
        raise ValueError(f"Expected at least two vehicle class folders under {data_dir}")
    class_indices = {name: idx for idx, name in enumerate(class_names)}
    recordings_by_class = {
        name: sorted(
            recording
            for folder in folders
            for recording in folder.iterdir()
            if recording.is_dir() and (recording / "ehz.csv").exists() and list(recording.glob("*.wav"))
        )
        for name, folders in class_dirs.items()
    }
    missing = [name for name, recordings in recordings_by_class.items() if not recordings]
    if missing:
        raise ValueError("Missing WAV + ehz.csv recordings for: " + ", ".join(missing))

    # Seeded run-level split. Classes with one recording use chronological blocks.
    run_split = split_runs(class_names, recordings_by_class, seed=42)
    output_dir = ROOT / "data" / "processed" / "MOD_vehicle"
    output_dir.mkdir(parents=True, exist_ok=True)
    if os.getenv("FOCAL_REBUILD_DATA") == "1":
        for stale in output_dir.glob("*.pt"):
            stale.unlink()

    all_paths: dict[str, dict[str, list[Path]]] = {key: {name: [] for name in class_names} for key in ("train", "val", "test")}
    class_run_test_method = {}
    total = 0
    for class_name in class_names:
        recordings = recordings_by_class[class_name]
        train_runs = set(run_split["train"][class_name])
        val_runs = set(run_split["val"][class_name])
        test_runs = set(run_split["test"][class_name])
        if len(recordings) == 1:
            class_run_test_method[class_name] = "chronological_windows_within_single_run"
        else:
            class_run_test_method[class_name] = "held_out_recording"
        for run in recordings:
            samples, _ = extract_recording(class_indices[class_name], class_name, run, output_dir)
            sample_stride = 1 if len(recordings) == 1 else max(1, int(os.getenv("FOCAL_SAMPLE_STRIDE", "4")))
            samples = samples[::sample_stride]
            n = len(samples)
            total += n
            if run in train_runs:
                if len(recordings) == 1:
                    train_end = int(n * 0.70)
                    val_end = int(n * 0.85)
                    all_paths["train"][class_name].extend(samples[:train_end])
                    all_paths["val"][class_name].extend(samples[train_end:val_end])
                    all_paths["test"][class_name].extend(samples[val_end:])
                elif len(recordings) == 2:
                    train_end = int(n * 0.85)
                    all_paths["train"][class_name].extend(samples[:train_end])
                    all_paths["val"][class_name].extend(samples[train_end:])
                else:
                    all_paths["train"][class_name].extend(samples)
            elif run in val_runs:
                all_paths["val"][class_name].extend(samples)
            elif run in test_runs:
                all_paths["test"][class_name].extend(samples)

    index_dir = ROOT / "data" / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_paths = {}
    for split_name, split_classes in all_paths.items():
        rows = [str(path.resolve()) for cls in class_names for path in split_classes[cls]]
        if not rows:
            raise ValueError(f"No samples assigned to {split_name} split")
        index_path = index_dir / f"MOD_vehicle_{split_name}.txt"
        index_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        index_paths[split_name] = str(index_path.resolve()).replace("\\", "/")

    config = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    config["vehicle_classification"]["num_classes"] = len(class_names)
    config["vehicle_classification"]["class_names"] = class_names
    config["vehicle_classification"]["train_index_file"] = index_paths["train"]
    config["vehicle_classification"]["val_index_file"] = index_paths["val"]
    config["vehicle_classification"]["test_index_file"] = index_paths["test"]
    config["pretrain_index_file"] = index_paths["train"]
    config["FOCAL"]["pretrain_lr_scheduler"]["train_epochs"] = int(os.getenv("FOCAL_PRETRAIN_EPOCHS", "5"))
    config["FOCAL"]["finetune_lr_scheduler"]["train_epochs"] = int(os.getenv("FOCAL_FINETUNE_EPOCHS", "20"))
    config["FOCAL"]["random_augmenters"]["time_augmenters"] = ["permutation", "negation", "horizontal_flip", "scaling"]
    config["DeepSense"].update({
        "loc_mod_out_channels": 64,
        "loc_mod_conv_inter_layers": 2,
        "loc_out_channels": 64,
        "loc_conv_inter_layers": 2,
        "recurrent_dim": 96,
        "recurrent_layers": 1,
        "fc_dim": 128,
    })
    MODEL_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "models" / "labels.json").write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = {
        split_name: {name: len(paths) for name, paths in split_classes.items()}
        for split_name, split_classes in all_paths.items()
    }
    metadata = {
        "dataset_id": "MOD_vehicle",
        "dataset_root": str(data_dir.resolve()),
        "classes": class_names,
        "sample_count": total,
        "split_counts": counts,
        "evaluation_split_by_class": class_run_test_method,
        "preprocessing": {
            "audio_source": "16 kHz mono WAV or first column of aud*.csv, resampled to 8 kHz",
            "seismic_source": "first column of ehz.csv at 100 Hz",
            "window_seconds": WINDOW_SECONDS,
            "interval_seconds": INTERVAL_SECONDS,
        },
    }
    (ROOT / "models" / "training_data.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {total} windows from {sum(map(len, recordings_by_class.values()))} recordings; classes={len(class_names)}")
    print("Split counts:", json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
