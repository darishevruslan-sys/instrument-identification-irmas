"""Tests for OpenMIC overlap-9 loading and masked metrics."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from src import config
from src.evaluate_openmic import compute_masked_metrics
from src.openmic import discover_openmic_samples


class OpenMICTests(unittest.TestCase):
    """OpenMIC smoke checks."""

    @unittest.skipUnless(config.DEFAULT_OPENMIC_DIR.exists(), "OpenMIC data is not available")
    def test_openmic_overlap9_test_split_counts(self) -> None:
        samples = discover_openmic_samples(config.DEFAULT_OPENMIC_DIR, split="test")
        labels = np.stack([sample.labels for sample in samples], axis=0)

        self.assertEqual(len(samples), 3339)
        self.assertEqual(int((labels.sum(axis=1) > 0).sum()), 1818)
        self.assertEqual(int((labels.sum(axis=1) >= 2).sum()), 323)

    @unittest.skipUnless(config.DEFAULT_OPENMIC_DIR.exists(), "OpenMIC data is not available")
    def test_openmic_smoke_sample_shapes(self) -> None:
        samples = discover_openmic_samples(config.DEFAULT_OPENMIC_DIR, split="test", max_files=32)

        self.assertEqual(len(samples), 32)
        for sample in samples:
            self.assertTrue(Path(sample.path).exists())
            self.assertEqual(sample.labels.shape, (len(config.OPENMIC_OVERLAP_CLASS_CODES),))
            self.assertEqual(sample.label_mask.shape, (len(config.OPENMIC_OVERLAP_CLASS_CODES),))
            self.assertTrue(bool(sample.label_mask.any()))

    def test_masked_metrics_ignore_unknown_labels(self) -> None:
        y_true = np.zeros((2, len(config.OPENMIC_OVERLAP_CLASS_CODES)), dtype=np.int64)
        y_pred = np.zeros_like(y_true)
        label_mask = np.zeros_like(y_true, dtype=bool)

        y_true[0, 0] = 1
        y_pred[0, 0] = 1
        label_mask[0, 0] = True

        y_true[1, 1] = 1
        y_pred[1, 1] = 0
        label_mask[1, 1] = True

        y_pred[:, 2] = 1

        metrics, _, _, rows = compute_masked_metrics(
            y_true=y_true,
            y_pred=y_pred,
            label_mask=label_mask,
            threshold=0.5,
            include_groups=False,
        )

        self.assertAlmostEqual(float(metrics["precision"]), 1.0)
        self.assertAlmostEqual(float(metrics["recall"]), 0.5)
        self.assertAlmostEqual(float(metrics["micro_f1"]), 2 / 3)
        self.assertEqual(rows[2]["known_count"], 0)
        self.assertEqual(rows[2]["predicted_count_known"], 0)


if __name__ == "__main__":
    unittest.main()
