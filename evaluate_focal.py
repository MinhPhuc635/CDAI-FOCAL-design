"""Recording-level held-out evaluation matching the dashboard's window aggregation."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch

from focal_pipeline import METRICS_PATH, ROOT, load_focal_model, predict_windows


def main() -> None:
    labels = json.loads((ROOT / "models" / "labels.json").read_text(encoding="utf-8"))
    test_index = ROOT / "data" / "indexes" / "MOD_vehicle_test.txt"
    groups: dict[str, list[Path]] = defaultdict(list)
    for line in test_index.read_text(encoding="utf-8").splitlines():
        sample_path = Path(line.strip())
        recording_id = sample_path.stem.rsplit("__", 1)[0]
        groups[recording_id].append(sample_path)

    model, _ = load_focal_model("cpu")
    predictions = []
    truths = []
    per_recording = []
    for recording_id, paths in sorted(groups.items()):
        audios, seismic = [], []
        for path in paths:
            item = torch.load(path, map_location="cpu", weights_only=True)
            audios.append(item["data"]["shake"]["audio"])
            seismic.append(item["data"]["shake"]["seismic"])
        # Each prepared item stores one sample as [1, intervals, samples].
        # Preserve the channel dimension required by the UI inference pipeline.
        audio = torch.cat(audios, dim=0).unsqueeze(1)
        sei = torch.cat(seismic, dim=0).unsqueeze(1)
        probabilities = predict_windows(model, audio, sei)
        true_label = recording_id.split("__", 1)[0]
        predicted_label = labels[int(probabilities.argmax())]
        truths.append(true_label)
        predictions.append(predicted_label)
        per_recording.append({
            "recording": recording_id,
            "windows": len(paths),
            "true": true_label,
            "predicted": predicted_label,
            "confidence_percent": round(float(probabilities.max()) * 100, 2),
        })

    correct = sum(actual == predicted for actual, predicted in zip(truths, predictions))
    accuracy = correct / max(1, len(truths))
    per_class = {}
    for name in labels:
        class_rows = [row for row in per_recording if row["true"] == name]
        per_class[name] = {
            "correct": sum(row["true"] == row["predicted"] for row in class_rows),
            "recordings": len(class_rows),
        }

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    training_data = json.loads((ROOT / "models" / "training_data.json").read_text(encoding="utf-8"))
    independent_classes = {
        name for name, method in training_data["evaluation_split_by_class"].items()
        if method == "held_out_recording"
    }
    independent_rows = [row for row in per_recording if row["true"] in independent_classes]
    independent_correct = sum(row["true"] == row["predicted"] for row in independent_rows)
    window_accuracy_percent = round(float(metrics["test_accuracy"]) * 100, 2)
    metrics["test_window_accuracy_percent"] = window_accuracy_percent
    metrics["test_recording_accuracy"] = accuracy
    metrics["test_recording_accuracy_percent"] = round(accuracy * 100, 2)
    metrics["test_independent_recordings"] = len(independent_rows)
    metrics["test_independent_recording_accuracy"] = independent_correct / max(1, len(independent_rows))
    metrics["test_independent_recording_accuracy_percent"] = round(
        metrics["test_independent_recording_accuracy"] * 100, 2
    )
    # The app reports the better-powered window metric as model test accuracy.
    metrics["test_accuracy_percent"] = window_accuracy_percent
    metrics["test_recordings"] = len(truths)
    metrics["test_per_class"] = per_class
    metrics["test_predictions"] = per_recording
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Recording-level held-out accuracy: {correct}/{len(truths)} = {accuracy * 100:.2f}%")
    for row in per_recording:
        print(f"{row['recording']}: {row['predicted']} ({row['confidence_percent']:.1f}%), expected {row['true']}")


if __name__ == "__main__":
    main()
