"""Dataset helpers for IRMAS Training Data."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
import warnings

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src import config
from src.features import empty_feature, extract_feature


@dataclass(frozen=True)
class AudioSample:
    """One audio file and its primary IRMAS class code."""

    path: Path
    class_code: str


def make_label_vector(class_code: str) -> np.ndarray:
    """Create a multi-hot vector for one IRMAS class code."""
    label = np.zeros(config.NUM_CLASSES, dtype=np.float32)
    label[config.CLASS_TO_INDEX[class_code]] = 1.0
    return label


def _safe_cache_name(text: str) -> str:
    """Create a stable short filename for a cache key."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _list_class_files(class_dir: Path) -> list[Path]:
    """List wav files in one class directory."""
    files = [path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"]
    return sorted(files)


def _limit_samples_balanced(samples_by_class: dict[str, list[AudioSample]], max_files: int) -> list[AudioSample]:
    """Limit samples with a simple round-robin pass across classes."""
    selected: list[AudioSample] = []
    class_codes = [class_code for class_code in config.CLASS_CODES if samples_by_class.get(class_code)]
    position = 0

    while len(selected) < max_files:
        added_any = False
        for class_code in class_codes:
            class_samples = samples_by_class[class_code]
            if position < len(class_samples):
                selected.append(class_samples[position])
                added_any = True
                if len(selected) >= max_files:
                    break
        if not added_any:
            break
        position += 1

    return selected


def discover_samples(data_dir: str | Path, max_files: int | None = None) -> list[AudioSample]:
    """Find IRMAS wav files and optionally limit their number for debug runs."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    samples_by_class: dict[str, list[AudioSample]] = {}
    for class_code in config.CLASS_CODES:
        class_dir = data_dir / class_code
        if not class_dir.exists():
            warnings.warn(f"IRMAS class directory is missing: {class_dir}", stacklevel=2)
            samples_by_class[class_code] = []
            continue

        samples_by_class[class_code] = [
            AudioSample(path=file_path, class_code=class_code) for file_path in _list_class_files(class_dir)
        ]

    samples = [sample for class_code in config.CLASS_CODES for sample in samples_by_class[class_code]]
    if not samples:
        raise ValueError(f"No .wav files found in {data_dir}")

    if max_files is not None:
        if max_files <= 0:
            raise ValueError("--max-files must be a positive integer")
        samples = _limit_samples_balanced(samples_by_class, max_files)

    return samples


def _can_stratify(samples: list[AudioSample], validation_size: float) -> bool:
    """Check whether sklearn stratification is possible for this split."""
    class_counts = Counter(sample.class_code for sample in samples)
    if len(class_counts) < 2 or min(class_counts.values()) < 2:
        return False

    val_count = int(round(len(samples) * validation_size))
    train_count = len(samples) - val_count
    num_classes = len(class_counts)
    return val_count >= num_classes and train_count >= num_classes


def split_samples(
    samples: list[AudioSample],
    validation_size: float = config.VALIDATION_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[list[AudioSample], list[AudioSample]]:
    """Split samples into train and validation sets."""
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1")
    if len(samples) < 2:
        raise ValueError("At least two samples are required for train/validation split")

    stratify = [sample.class_code for sample in samples] if _can_stratify(samples, validation_size) else None
    train_samples, val_samples = train_test_split(
        samples,
        test_size=validation_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )
    return list(train_samples), list(val_samples)


class IRMASDataset(Dataset):
    """PyTorch dataset for IRMAS spectral features and multi-hot labels."""

    def __init__(
        self,
        samples: list[AudioSample],
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
        augment: bool = False,
        augment_probability: float = 0.8,
        noise_std: float = 0.03,
        gain_min: float = 0.85,
        gain_max: float = 1.15,
        max_time_shift: int = 8,
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
        self.augment = augment
        self.augment_probability = augment_probability
        self.noise_std = noise_std
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.max_time_shift = max_time_shift

    def _cache_path(self, sample: AudioSample) -> Path:
        """Return the .npy cache path for one sample and feature configuration."""
        try:
            relative_path = sample.path.resolve().relative_to(config.PROJECT_ROOT.resolve())
        except ValueError:
            relative_path = sample.path.resolve()

        cache_key = "|".join(
            [
                str(relative_path).replace("\\", "/"),
                self.feature_type,
                str(self.sample_rate),
                str(self.duration),
                str(self.n_fft),
                str(self.hop_length),
                str(self.n_mels),
                str(self.n_mfcc),
            ]
        )
        filename = f"{sample.class_code}_{_safe_cache_name(cache_key)}.npy"
        return self.cache_dir / self.feature_type / filename

    def _load_or_extract_feature(self, sample: AudioSample) -> np.ndarray:
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

    def _augment_feature(self, feature: np.ndarray) -> np.ndarray:
        """Apply light feature-level augmentation for training."""
        if not self.augment or np.random.random() > self.augment_probability:
            return feature

        augmented = feature.copy()

        gain = np.random.uniform(self.gain_min, self.gain_max)
        augmented = augmented * gain

        if self.noise_std > 0:
            noise = np.random.normal(0.0, self.noise_std, size=augmented.shape)
            augmented = augmented + noise.astype(np.float32)

        if self.max_time_shift > 0:
            shift = int(np.random.randint(-self.max_time_shift, self.max_time_shift + 1))
            if shift != 0:
                shifted = np.zeros_like(augmented)
                if shift > 0:
                    shifted[:, :, shift:] = augmented[:, :, :-shift]
                else:
                    shifted[:, :, :shift] = augmented[:, :, -shift:]
                augmented = shifted

        return augmented.astype(np.float32)

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

        feature = self._augment_feature(feature)

        return {
            "features": torch.from_numpy(feature),
            "labels": torch.from_numpy(make_label_vector(sample.class_code)),
            "path": str(sample.path),
            "class_code": sample.class_code,
        }


def class_distribution(samples: list[AudioSample]) -> dict[str, int]:
    """Return sample counts by class code."""
    counts = Counter(sample.class_code for sample in samples)
    return {class_code: counts.get(class_code, 0) for class_code in config.CLASS_CODES}
