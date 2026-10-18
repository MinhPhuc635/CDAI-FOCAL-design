"""Build the FOCAL self-supervised checkpoint, fine-tune it, and score the held-out data."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "focal-main" / "src"
PYTHON = sys.executable
MODEL_NAME = "DeepSense"
# FOCAL ships its vehicle classification schema as MOD.yaml; the raw source
# data for this project is MOD_vehicle, selected independently via its path.
FOCAL_CONFIG_NAME = "MOD"
WEIGHTS = SRC.parent / "weights" / f"MOD_{MODEL_NAME}"
LOG_PATH = ROOT / "models" / "training.log"


def run_logged(command: list[str], cwd: Path, env: dict[str, str], log) -> None:
    print("$", " ".join(command), flush=True)
    log.write("$ " + " ".join(command) + "\n")
    log.flush()
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    if process.wait() != 0:
        raise RuntimeError(f"FOCAL command failed; see {LOG_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rebuild the model even when a checkpoint exists")
    args_cli = parser.parse_args()
    model_path = ROOT / "models" / "vehicle_focal.pt"
    metrics_path = ROOT / "models" / "model_metrics.json"
    env = os.environ.copy()
    env.setdefault("MOD_VEHICLE_DATA_DIR", r"F:\MOD_vehicle\raw_data")
    dataset_root = Path(env["MOD_VEHICLE_DATA_DIR"]).resolve()
    try:
        current_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        current_metrics = {}
    checkpoint_matches_dataset = current_metrics.get("dataset_id") == "MOD_vehicle"
    if model_path.exists() and metrics_path.exists() and checkpoint_matches_dataset and not args_cli.force:
        print(f"Existing checkpoint ready: {model_path}")
        return
    if not SRC.is_dir():
        raise FileNotFoundError(f"FOCAL source folder not found: {SRC}")
    (ROOT / "models").mkdir(exist_ok=True)

    env["OMP_NUM_THREADS"] = env.get("FOCAL_CPU_THREADS", "8")
    env["MKL_NUM_THREADS"] = env["OMP_NUM_THREADS"]
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env.setdefault("FOCAL_PRETRAIN_EPOCHS", "5")
    env.setdefault("FOCAL_FINETUNE_EPOCHS", "20")
    # FOCAL reads epoch counts from MOD.yaml, not from environment variables.
    # Keep this config in sync even when the prepared tensor cache is reused.
    config_path = SRC / "data" / f"{FOCAL_CONFIG_NAME}.yaml"
    focal_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    focal_config["FOCAL"]["pretrain_lr_scheduler"]["train_epochs"] = int(env["FOCAL_PRETRAIN_EPOCHS"])
    focal_config["FOCAL"]["finetune_lr_scheduler"]["train_epochs"] = int(env["FOCAL_FINETUNE_EPOCHS"])
    config_path.write_text(yaml.safe_dump(focal_config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    with LOG_PATH.open("w", encoding="utf-8") as log:
        data_ready = all((ROOT / "data" / "indexes" / f"MOD_vehicle_{part}.txt").exists()
                         for part in ("train", "val", "test"))
        training_data_path = ROOT / "models" / "training_data.json"
        try:
            prepared_data = json.loads(training_data_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prepared_data = {}
        data_ready = data_ready and prepared_data.get("dataset_id") == "MOD_vehicle"
        data_ready = data_ready and prepared_data.get("dataset_root") == str(dataset_root)
        if args_cli.force or not data_ready:
            env["FOCAL_REBUILD_DATA"] = "1"
        if not data_ready or args_cli.force:
            run_logged([PYTHON, str(ROOT / "prepare_mod_data.py")], ROOT, env, log)
        else:
            print("Using the existing prepared MOD window indexes.", flush=True)
            log.write("Using the existing prepared MOD window indexes.\n")
        run_logged([PYTHON, "train.py", f"-model={MODEL_NAME}", f"-dataset={FOCAL_CONFIG_NAME}", "-learn_framework=FOCAL",
                    "-stage=pretrain", "-batch_size=32", "-gpu=cpu"], SRC, env, log)

        pretrain_folders = sorted(WEIGHTS.glob("exp*_contrastive_FOCAL"), key=lambda p: p.stat().st_mtime)
        if not pretrain_folders:
            raise FileNotFoundError(f"FOCAL did not produce a pretraining folder under {WEIGHTS}")
        pretrain_folder = pretrain_folders[-1]
        pretrain_weight = pretrain_folder / f"MOD_{MODEL_NAME}_pretrain_latest.pt"
        if not pretrain_weight.exists():
            raise FileNotFoundError(f"FOCAL pretraining checkpoint was not created: {pretrain_weight}")

        run_logged([PYTHON, "train.py", f"-model={MODEL_NAME}", f"-dataset={FOCAL_CONFIG_NAME}", "-learn_framework=FOCAL",
                    "-stage=finetune", "-task=vehicle_classification", "-batch_size=64", "-gpu=cpu",
                    f"-model_weight={pretrain_folder}"], SRC, env, log)
        final_weight = pretrain_folder / f"MOD_{MODEL_NAME}_vehicle_classification_1.0_finetune_best.pt"
        if not final_weight.exists():
            raise FileNotFoundError(f"Fine-tuned FOCAL checkpoint was not created: {final_weight}")
        output = subprocess.run(
            [PYTHON, "test.py", f"-model={MODEL_NAME}", f"-dataset={FOCAL_CONFIG_NAME}", "-learn_framework=FOCAL",
             "-stage=finetune", "-task=vehicle_classification", "-batch_size=64", "-gpu=cpu",
             f"-model_weight={pretrain_folder}"],
            cwd=SRC, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
        )
        log.write("\n[Final held-out evaluation]\n" + output.stdout + output.stderr)
        metrics_match = re.search(r"Test acc:\s*([0-9.]+),\s*test f1:\s*([0-9.]+)", output.stdout)
        if not metrics_match:
            raise RuntimeError("Could not read held-out accuracy from FOCAL test output; see training.log")

        shutil.copy2(final_weight, model_path)
        training_data = json.loads((ROOT / "models" / "training_data.json").read_text(encoding="utf-8"))
        metrics = {
            "dataset_id": "MOD_vehicle",
            "dataset_root": str(dataset_root),
            "test_accuracy": float(metrics_match.group(1)),
            "test_macro_f1": float(metrics_match.group(2)),
            "test_accuracy_percent": round(float(metrics_match.group(1)) * 100, 2),
            "classes": training_data["classes"],
            "sample_count": training_data["sample_count"],
            "split_counts": training_data["split_counts"],
            "evaluation_split_by_class": training_data["evaluation_split_by_class"],
            "checkpoint": model_path.name,
            "architecture": f"FOCAL pretraining + {MODEL_NAME} fine-tuning",
        }
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nCheckpoint saved to {model_path}")
        print(f"Held-out accuracy: {metrics['test_accuracy_percent']:.2f}%")


if __name__ == "__main__":
    main()
