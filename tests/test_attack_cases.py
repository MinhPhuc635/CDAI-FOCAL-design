"""Ten low-impact security checks against the local API only.

These do not scan or load-test any public service. Run with:
python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

import api


class SafeLocalAttackCases(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    @staticmethod
    def pair(audio_name="aud.csv", audio=b"0.1\n0.2\n", seismic=b"0.1\n0.2\n"):
        return {
            "audio": (audio_name, audio, "application/octet-stream"),
            "seismic": ("ehz.csv", seismic, "text/csv"),
        }

    def test_a01_fake_wav_extension_does_not_bypass_file_parser(self):
        response = self.client.post("/api/v1/predict", files=self.pair("vehicle.wav", b"MZ-not-a-wave-file"))
        self.assertEqual(response.status_code, 422)

    def test_a02_csv_formula_string_is_treated_as_invalid_signal(self):
        payload = b'=HYPERLINK("https://example.invalid","open")\n'
        response = self.client.post("/api/v1/predict", files=self.pair(audio=payload))
        self.assertEqual(response.status_code, 422)

    def test_a03_nan_and_infinity_are_rejected(self):
        response = self.client.post("/api/v1/predict", files=self.pair(audio=b"NaN\nInf\n"))
        self.assertEqual(response.status_code, 422)

    def test_a04_traversal_filename_is_not_echoed_or_used_as_a_path(self):
        with (
            patch.object(api, "_model_and_labels", return_value=(object(), ["car", "bus"])),
            patch.object(api.focal, "read_audio_csv", return_value=np.ones(32000, dtype=np.float32)),
            patch.object(api.focal, "read_seismic_csv", return_value=np.ones(200, dtype=np.float32)),
            patch.object(api.focal, "make_time_windows", return_value=(np.zeros(1), np.zeros(1))),
            patch.object(api.focal, "predict_windows", return_value=np.array([0.9, 0.1])),
            patch.object(api.focal, "detect_out_of_distribution_signal", return_value=None),
        ):
            response = self.client.post("/api/v1/predict", files=self.pair("..\\..\\private\\aud.csv"))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotIn("private", response.text.lower())
        self.assertNotIn("filename", response.json())

    def test_a05_single_upload_size_bomb_is_rejected(self):
        response = self.client.post(
            "/api/v1/predict", files=self.pair(audio=b"x" * (16 * 1024 * 1024 + 1))
        )
        self.assertEqual(response.status_code, 413)

    def test_a06_combined_upload_size_bomb_is_rejected(self):
        response = self.client.post(
            "/api/v1/predict",
            files=self.pair(audio=b"x" * (13 * 1024 * 1024), seismic=b"y" * (12 * 1024 * 1024)),
        )
        self.assertEqual(response.status_code, 413)

    def test_a07_authentication_bypass_without_key_is_blocked_when_key_configured(self):
        with patch.dict(os.environ, {"FOCAL_API_KEY": "only-for-this-test"}):
            response = self.client.post("/api/v1/predict", files=self.pair())
        self.assertEqual(response.status_code, 401)

    def test_a08_bounded_repeat_health_requests_remain_available(self):
        # Five local requests are a smoke check, not a denial-of-service load test.
        responses = [self.client.get("/api/v1/health") for _ in range(5)]
        self.assertTrue(all(response.status_code == 200 for response in responses))

    def test_a09_broadband_out_of_scope_signal_is_rejected_as_unknown(self):
        rng = np.random.default_rng(42)
        audio_noise = rng.normal(size=32_000).astype(np.float32)
        seismic_noise = rng.normal(size=200).astype(np.float32)
        with (
            patch.object(api, "_model_and_labels", return_value=(object(), ["car", "bus"])),
            patch.object(api.focal, "read_audio_csv", return_value=audio_noise),
            patch.object(api.focal, "read_seismic_csv", return_value=seismic_noise),
            patch.object(api.focal, "make_time_windows", return_value=(np.zeros(1), np.zeros(1))),
            patch.object(api.focal, "predict_windows", return_value=np.array([0.98, 0.02])),
        ):
            response = self.client.post("/api/v1/predict", files=self.pair())
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["unknown"])
        self.assertIsNone(response.json()["predicted_class"])

    def test_a10_checkpoint_failure_returns_generic_error_without_traceback(self):
        with (
            patch.object(api.focal, "read_audio_csv", return_value=np.ones(32000, dtype=np.float32)),
            patch.object(api.focal, "read_seismic_csv", return_value=np.ones(200, dtype=np.float32)),
            patch.object(api.focal, "make_time_windows", return_value=(np.zeros(1), np.zeros(1))),
            patch.object(api, "_model_and_labels", side_effect=RuntimeError("private checkpoint path leaked")),
            self.assertLogs("focal.api", level="ERROR"),
        ):
            response = self.client.post("/api/v1/predict", files=self.pair())
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private checkpoint path", response.text)
        self.assertNotIn("Traceback", response.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
