"""Ten API/system test cases for the FOCAL classifier.

Run: python -m unittest discover -s tests -v
The labeled-data AI case uses environment variables described in the PDF.
"""
from __future__ import annotations

import io
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

import api


class FocalApiSystemTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    @staticmethod
    def files(audio_name="aud.csv", audio=b"0.1\n0.2\n", seismic=b"0.1\n0.2\n"):
        return {
            "audio": (audio_name, audio, "application/octet-stream"),
            "seismic": ("ehz.csv", seismic, "text/csv"),
        }

    def test_01_health_reports_checkpoint_state(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json()["status"], {"ready", "needs_checkpoint"})
        self.assertEqual(response.json()["model"], "FOCAL")

    def test_02_missing_required_upload_is_rejected(self):
        response = self.client.post("/api/v1/predict", files={"audio": ("aud.csv", b"1", "text/csv")})
        self.assertEqual(response.status_code, 422)

    def test_03_wrong_file_extension_is_rejected(self):
        response = self.client.post("/api/v1/predict", files=self.files("audio.exe"))
        self.assertEqual(response.status_code, 415)

    def test_04_single_file_over_limit_is_rejected(self):
        response = self.client.post(
            "/api/v1/predict",
            files=self.files(audio=b"0" * (16 * 1024 * 1024 + 1)),
        )
        self.assertEqual(response.status_code, 413)

    def test_05_combined_upload_over_limit_is_rejected(self):
        response = self.client.post(
            "/api/v1/predict",
            files=self.files(audio=b"a" * (13 * 1024 * 1024), seismic=b"s" * (12 * 1024 * 1024)),
        )
        self.assertEqual(response.status_code, 413)

    def test_06_configured_api_key_blocks_anonymous_request(self):
        with patch.dict(os.environ, {"FOCAL_API_KEY": "test-secret"}):
            response = self.client.post("/api/v1/predict", files=self.files())
        self.assertEqual(response.status_code, 401)

    def test_07_mocked_inference_returns_prediction_contract(self):
        with (
            patch.object(api, "_model_and_labels", return_value=(object(), ["bicycle", "car"])),
            patch.object(api.focal, "read_audio_csv", return_value=np.ones(32000, dtype=np.float32)),
            patch.object(api.focal, "read_seismic_csv", return_value=np.ones(200, dtype=np.float32)),
            patch.object(api.focal, "make_time_windows", return_value=(np.zeros(1), np.zeros(1))),
            patch.object(api.focal, "predict_windows", return_value=np.array([0.9, 0.1])),
            patch.object(api.focal, "detect_out_of_distribution_signal", return_value=None),
        ):
            response = self.client.post("/api/v1/predict", files=self.files())
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["predicted_class"], "bicycle")
        self.assertAlmostEqual(result["confidence"], 0.9)
        self.assertEqual(len(result["classes"]), 2)

    def test_08_invalid_signal_is_returned_as_client_error(self):
        with patch.object(api.focal, "read_audio_csv", side_effect=ValueError("invalid signal")):
            response = self.client.post("/api/v1/predict", files=self.files())
        self.assertEqual(response.status_code, 422)
        self.assertIn("invalid signal", response.json()["detail"])

    def test_09_missing_checkpoint_returns_service_unavailable(self):
        with (
            patch.object(api.focal, "read_audio_csv", return_value=np.ones(32000, dtype=np.float32)),
            patch.object(api.focal, "read_seismic_csv", return_value=np.ones(200, dtype=np.float32)),
            patch.object(api.focal, "make_time_windows", return_value=(np.zeros(1), np.zeros(1))),
            patch.object(api, "_model_and_labels", side_effect=FileNotFoundError("checkpoint missing")),
        ):
            response = self.client.post("/api/v1/predict", files=self.files())
        self.assertEqual(response.status_code, 503)

    def test_10_real_recording_through_focal_model(self):
        audio_path = os.getenv("FOCAL_TEST_AUDIO_FILE")
        seismic_path = os.getenv("FOCAL_TEST_SEISMIC_FILE")
        expected_label = os.getenv("FOCAL_TEST_EXPECTED_LABEL")
        if not (audio_path and seismic_path):
            self.skipTest("Set FOCAL_TEST_AUDIO_FILE and FOCAL_TEST_SEISMIC_FILE to a real synchronized pair.")
        audio_file, seismic_file = Path(audio_path), Path(seismic_path)
        if not audio_file.is_file() or not seismic_file.is_file():
            self.skipTest("Configured Kaggle sample files are not present yet.")
        response = self.client.post(
            "/api/v1/predict",
            files={
                "audio": (audio_file.name, audio_file.read_bytes(), "application/octet-stream"),
                "seismic": (seismic_file.name, seismic_file.read_bytes(), "text/csv"),
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["classes"][0]["label"], result["nearest_class"])
        values = [item["probability"] for item in result["classes"]]
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values))
        self.assertEqual(values, sorted(values, reverse=True))
        if expected_label:
            self.assertEqual(result["nearest_class"], expected_label)
        else:
            self.assertTrue(result["nearest_class"], "FOCAL should return its nearest class even when ground truth is unknown.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
