"""OpenMIC 2018 dataset helpers for external multi-label evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from src import config
from src.features import empty_feature, extract_feature


@dataclass(frozen=True)
class OpenMICSample:
    """One OpenMIC audio clip with overlap labels and known-label mask."""

    sample_key: str
    path: Path
    labels: np.ndarray
    label_mask: np.ndarray


def _safe_cache_name(text: str) -> str:
    """Create a stable short filename for a cache key."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def audio_path_for_sample(data_dir: str | Path, sample_key: str) -> Path:
    """Return the expected OpenMIC .ogg path for one sample key."""
    data_dir = Path(data_dir)
    return data_dir / "audio" / sample_key[:3] / f"{sample_key}.ogg"


def load_class_map(data_dir: str | Path) -> dict[str, int]:
    """Load OpenMIC instrument name to column index mapping."""
    class_map_path = Path(data_dir) / "class-map.json"
    if not class_map_path.exists():
        raise FileNotFoundError(f"OpenMIC class map not found: {class_map_path}")
    return json.loads(class_map_path.read_text(encoding="utf-8"))


def load_partition_keys(data_dir: str | Path, split: str = "test") -> list[str]:
    """Load sample keys for one official OpenMIC split."""
    data_dir = Path(data_dir)
    split_file = split
    if split in {"train", "test"}:
        split_file = f"split01_{split}.csv"

    partition_path = data_dir / "partitions" / split_file
    if not partition_path.exists():
        raise FileNotFoundError(f"OpenMIC partition file not found: {partition_path}")

    return [
        line.strip()
        for line in partition_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def discover_openmic_samples(
    data_dir: str | Path = config.DEFAULT_OPENMIC_DIR,
    split: str = "test",
    max_files: int | None = None,
    positive_threshold: float = 0.5,
    require_known_overlap: bool = True,
) -> list[OpenMICSample]:
    """Load OpenMIC overlap-9 samples from the official split."""
    data_dir = Path(data_dir)
    npz_path = data_dir / "openmic-2018.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"OpenMIC npz file not found: {npz_path}")
    if max_files is not None and max_files <= 0:
        raise ValueError("--max-files must be a positive integer")

    class_map = load_class_map(data_dir)
    missing_classes = [
        class_name
        for class_name in config.OPENMIC_OVERLAP_CLASS_NAMES
        if class_name not in class_map
    ]
    if missing_classes:
        raise ValueError(f"OpenMIC class map is missing classes: {missing_classes}")

    openmic_indices = [class_map[class_name] for class_name in config.OPENMIC_OVERLAP_CLASS_NAMES]
    data = np.load(npz_path, allow_pickle=True)
    sample_keys = data["sample_key"].astype(str)
    key_to_row = {sample_key: index for index, sample_key in enumerate(sample_keys)}
    y_true = data["Y_true"]
    y_mask = data["Y_mask"].astype(bool)

    samples: list[OpenMICSample] = []
    for sample_key in load_partition_keys(data_dir, split=split):
        if sample_key not in key_to_row:
            warnings.warn(f"OpenMIC split key is missing in npz: {sample_key}", stacklevel=2)
            continue

        row_index = key_to_row[sample_key]
        label_mask = y_mask[row_index, openmic_indices].astype(bool)
        if require_known_overlap and not bool(label_mask.any()):
            continue

        relevance = y_true[row_index, openmic_indices]
        labels = ((relevance > positive_threshold) & label_mask).astype(np.float32)
        audio_path = audio_path_for_sample(data_dir, sample_key)
        if not audio_path.exists():
            raise FileNotFoundError(f"OpenMIC audio file not found: {audio_path}")

        samples.append(
            OpenMICSample(
                sample_key=sample_key,
                path=audio_path,
                labels=labels,
                label_mask=label_mask,
            )
        )
        if max_files is not None and len(samples) >= max_files:
            break

    if not samples:
        raise ValueError(f"No OpenMIC samples found for split={split} in {data_dir}")

    return samples


def openmic_distribution(samples: list[OpenMICSample]) -> dict[str, int]:
    """Return positive sample counts for overlap-9 classes."""
    labels = np.stack([sample.labels for sample in samples], axis=0)
    return {
        class_code: int(labels[:, index].sum())
        for index, class_code in enumerate(config.OPENMIC_OVERLAP_CLASS_CODES)
    }


class OpenMICDataset(Dataset):
    """PyTorch dataset for OpenMIC spectral features, labels and label masks."""

    def __init__(
        self,
        samples: list[OpenMICSample],
        feature_type: str = config.DEFAULT_FEATURE_TYPE,
        sample_rate: int = config.SAMPLE_RATE,
        duration: float = config.DURATION,
        n_fft: int = config.N_FFT,
        hop_length: int = config.HOP_LENGTH,
        n_mels: int = config.N_MELS,
        n_mfcc: int = config.N_MFCC,
        ignore_errors: bool = True,
        use_cache: bool = True,
        cache_dir: str | Path = config.FEATURE_CACHE_DIR,
    ) -> None:
        if feature_type not in config.FEATURE_TYPES:
            raise ValueError(f"feature_type must be one of {config.FEATURE_TYPES}")

        self.samples = samples
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.ignore_errors = ignore_errors
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)

    def _cache_path(self, sample: OpenMICSample) -> Path:
        """Return the .npy cache path for one sample and feature configuration."""
        try:
            relative_path = sample.path.resolve().relative_to(config.PROJECT_ROOT.resolve())
        except ValueError:
            relative_path = sample.path.resolve()

        cache_key = "|".join(
            [
                str(relative_path).replace("\\", "/"),
                sample.sample_key,
                self.feature_type,
                str(self.sample_rate),
                str(self.duration),
                str(self.n_fft),
                str(self.hop_length),
                str(self.n_mels),
                str(self.n_mfcc),
            ]
        )
        filename = f"{sample.sample_key}_{_safe_cache_name(cache_key)}.npy"
        return self.cache_dir / "openmic" / self.feature_type / filename

    def _load_or_extract_feature(self, sample: OpenMICSample) -> np.ndarray:
        """Load a cached feature if available, otherwise extract and cache it."""
        if self.use_cache:
            cache_path = self._cache_path(sample)
            if cache_path.exists():
                return np.load(cache_path).astype(np.float32)

        feature = extract_feature(
            sample.path,
            feature_type=self.feature_type,
            sample_rate=self.sample_rate,
            duration=self.duration,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            n_mfcc=self.n_mfcc,
        )

        if self.use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, feature)

        return feature

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        try:
            feature = self._load_or_extract_feature(sample)
        except Exception as exc:
            if not self.ignore_errors:
                raise
            warnings.warn(f"Could not process audio file {sample.path}: {exc}", stacklevel=2)
            feature = empty_feature(
                feature_type=self.feature_type,
                sample_rate=self.sample_rate,
                duration=self.duration,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                n_mfcc=self.n_mfcc,
            )

        return {
            "features": torch.from_numpy(feature),
            "labels": torch.from_numpy(sample.labels.astype(np.float32)),
            "label_mask": torch.from_numpy(sample.label_mask.astype(bool)),
            "path": str(sample.path),
            "sample_key": sample.sample_key,
        }
