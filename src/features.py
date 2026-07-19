"""Audio loading and spectral feature extraction."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from src import config


def target_num_samples(sample_rate: int = config.SAMPLE_RATE, duration: float = config.DURATION) -> int:
    """Return the fixed number of samples for one audio fragment."""
    return int(round(sample_rate * duration))


def expected_time_frames(
    sample_rate: int = config.SAMPLE_RATE,
    duration: float = config.DURATION,
    hop_length: int = config.HOP_LENGTH,
) -> int:
    """Return the expected number of frames for librosa features with center=True."""
    return 1 + target_num_samples(sample_rate, duration) // hop_length


def get_feature_shape(
    feature_type: str,
    sample_rate: int = config.SAMPLE_RATE,
    duration: float = config.DURATION,
    hop_length: int = config.HOP_LENGTH,
    n_mels: int = config.N_MELS,
    n_mfcc: int = config.N_MFCC,
) -> tuple[int, int, int]:
    """Return the model input shape without batch dimension."""
    time_frames = expected_time_frames(sample_rate, duration, hop_length)
    if feature_type == "mel":
        return (1, n_mels, time_frames)
    if feature_type == "mfcc":
        return (1, n_mfcc, time_frames)
    raise ValueError(f"Unsupported feature_type: {feature_type}")


def load_audio(
    file_path: str | Path,
    sample_rate: int = config.SAMPLE_RATE,
    duration: float = config.DURATION,
    mono: bool = config.MONO,
) -> np.ndarray:
    """Load a wav file and crop or pad it to a fixed duration."""
    file_path = Path(file_path)
    audio, _ = librosa.load(file_path, sr=sample_rate, mono=mono)
    target_length = target_num_samples(sample_rate, duration)

    if len(audio) > target_length:
        audio = audio[:target_length]
    elif len(audio) < target_length:
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), mode="constant")

    return audio.astype(np.float32)


def normalize_feature(feature: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a feature matrix by mean and standard deviation."""
    mean = float(feature.mean())
    std = float(feature.std())
    if std < eps:
        return (feature - mean).astype(np.float32)
    return ((feature - mean) / (std + eps)).astype(np.float32)


def fix_time_dimension(feature: np.ndarray, expected_frames: int) -> np.ndarray:
    """Pad or crop the time axis so all features have identical shape."""
    current_frames = feature.shape[1]
    if current_frames > expected_frames:
        return feature[:, :expected_frames]
    if current_frames < expected_frames:
        pad_width = expected_frames - current_frames
        return np.pad(feature, ((0, 0), (0, pad_width)), mode="constant")
    return feature


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = config.SAMPLE_RATE,
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH,
    n_mels: int = config.N_MELS,
    duration: float = config.DURATION,
) -> np.ndarray:
    """Compute a normalized log Mel-spectrogram with shape [1, n_mels, T]."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = fix_time_dimension(
        mel_db,
        expected_time_frames(sample_rate=sample_rate, duration=duration, hop_length=hop_length),
    )
    mel_db = normalize_feature(mel_db)
    return np.expand_dims(mel_db, axis=0).astype(np.float32)


def compute_mfcc(
    audio: np.ndarray,
    sample_rate: int = config.SAMPLE_RATE,
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH,
    n_mfcc: int = config.N_MFCC,
    duration: float = config.DURATION,
) -> np.ndarray:
    """Compute normalized MFCC features with shape [1, n_mfcc, T]."""
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    mfcc = fix_time_dimension(
        mfcc,
        expected_time_frames(sample_rate=sample_rate, duration=duration, hop_length=hop_length),
    )
    mfcc = normalize_feature(mfcc)
    return np.expand_dims(mfcc, axis=0).astype(np.float32)


def extract_feature(
    file_path: str | Path,
    feature_type: str = config.DEFAULT_FEATURE_TYPE,
    sample_rate: int = config.SAMPLE_RATE,
    duration: float = config.DURATION,
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH,
    n_mels: int = config.N_MELS,
    n_mfcc: int = config.N_MFCC,
) -> np.ndarray:
    """Load an audio file and extract the selected spectral feature."""
    audio = load_audio(file_path, sample_rate=sample_rate, duration=duration, mono=config.MONO)

    if feature_type == "mel":
        return compute_mel_spectrogram(
            audio,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            duration=duration,
        )
    if feature_type == "mfcc":
        return compute_mfcc(
            audio,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mfcc=n_mfcc,
            duration=duration,
        )
    raise ValueError(f"Unsupported feature_type: {feature_type}")


def empty_feature(
    feature_type: str = config.DEFAULT_FEATURE_TYPE,
    sample_rate: int = config.SAMPLE_RATE,
    duration: float = config.DURATION,
    hop_length: int = config.HOP_LENGTH,
    n_mels: int = config.N_MELS,
    n_mfcc: int = config.N_MFCC,
) -> np.ndarray:
    """Return a zero feature tensor for unreadable audio files."""
    shape = get_feature_shape(
        feature_type=feature_type,
        sample_rate=sample_rate,
        duration=duration,
        hop_length=hop_length,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
    )
    return np.zeros(shape, dtype=np.float32)
