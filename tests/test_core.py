"""Fast deterministic checks that do not require the datasets."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch

from src import config
from src.dataset import AudioSample, split_samples
from src.features import compute_mel_spectrogram, get_feature_shape
from src.model import build_model
from src.utils import get_state_dict_from_checkpoint, load_checkpoint


class CoreSmokeTests(unittest.TestCase):
    def test_split_is_reproducible_with_explicit_seed(self) -> None:
        samples = [
            AudioSample(Path(f"{code}_{index}.wav"), code)
            for code in config.CLASS_CODES
            for index in range(5)
        ]
        first_train, first_val = split_samples(samples, validation_size=0.4, random_state=42)
        second_train, second_val = split_samples(samples, validation_size=0.4, random_state=42)

        self.assertEqual([item.path for item in first_train], [item.path for item in second_train])
        self.assertEqual([item.path for item in first_val], [item.path for item in second_val])

    def test_synthetic_audio_has_expected_mel_shape(self) -> None:
        samples = int(config.SAMPLE_RATE * config.DURATION)
        time = np.arange(samples, dtype=np.float32) / config.SAMPLE_RATE
        audio = np.sin(2 * np.pi * 440.0 * time).astype(np.float32)
        feature = compute_mel_spectrogram(audio)
        self.assertEqual(feature.shape, get_feature_shape("mel"))
        self.assertTrue(np.isfinite(feature).all())

    def test_public_checkpoint_loads_and_runs_inference(self) -> None:
        checkpoint_path = config.MODELS_DIR / "best_model_mel_v2.pth"
        self.assertTrue(checkpoint_path.exists())
        checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
        model = build_model(num_classes=config.NUM_CLASSES, model_size="medium")
        model.load_state_dict(get_state_dict_from_checkpoint(checkpoint))
        model.eval()

        with torch.no_grad():
            logits = model(torch.zeros((1, *get_feature_shape("mel")), dtype=torch.float32))
        self.assertEqual(tuple(logits.shape), (1, config.NUM_CLASSES))
        self.assertTrue(bool(torch.isfinite(logits).all()))


if __name__ == "__main__":
    unittest.main()
