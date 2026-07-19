"""Upload validation and API smoke tests."""

from __future__ import annotations

import io
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from web import backend


class UploadValidationTests(unittest.TestCase):
    @staticmethod
    def _synthetic_wav() -> bytes:
        buffer = io.BytesIO()
        sample_rate = 22_050
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        audio = (0.1 * np.sin(2 * np.pi * 440.0 * time)).astype(np.float32)
        sf.write(buffer, audio, sample_rate, format="WAV")
        return buffer.getvalue()

    def test_unsupported_extension_returns_400(self) -> None:
        upload = UploadFile(filename="payload.exe", file=io.BytesIO(b"not audio"))
        with self.assertRaises(HTTPException) as raised:
            backend._read_upload(upload)
        self.assertEqual(raised.exception.status_code, 400)

    def test_size_limit_returns_413(self) -> None:
        upload = UploadFile(filename="large.wav", file=io.BytesIO(b"x" * 11))
        with patch.object(backend, "MAX_UPLOAD_BYTES", 10):
            with self.assertRaises(HTTPException) as raised:
                backend._read_upload(upload)
        self.assertEqual(raised.exception.status_code, 413)

    def test_model_info_is_upload_only_and_ready(self) -> None:
        with TestClient(backend.app) as client:
            response = client.get("/api/model-info")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ready"])
        self.assertEqual(payload["inference_mode"], "upload_only")
        self.assertEqual(payload["max_upload_mb"], 40)
        self.assertEqual(payload["checkpoint"], "best_model_mel_v2.pth")
        self.assertNotIn("tracks", payload)
        self.assertNotIn("tracks_count", payload)

    def test_analyze_accepts_valid_synthetic_audio(self) -> None:
        with TestClient(backend.app) as client:
            response = client.post(
                "/api/analyze",
                files={"file": ("tone.wav", self._synthetic_wav(), "audio/wav")},
            )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(len(payload["classes"]), 11)
        self.assertEqual(payload["model"]["input_shape"], [1, 128, 130])

    def test_analyze_rejects_unsupported_extension(self) -> None:
        with TestClient(backend.app) as client:
            response = client.post(
                "/api/analyze",
                files={"file": ("payload.exe", b"not audio", "application/octet-stream")},
            )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
