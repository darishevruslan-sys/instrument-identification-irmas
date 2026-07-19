"""Utility functions for experiments and file outputs."""

from __future__ import annotations

from pathlib import Path
import json
import random

import numpy as np
import torch

from src import config


def set_seed(seed: int = config.RANDOM_STATE) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return a PyTorch device, falling back to CPU when GPU is unavailable."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_output_dirs() -> None:
    """Create standard output directories."""
    for path in [
        config.FIGURES_DIR,
        config.METRICS_DIR,
        config.MODELS_DIR,
        config.PREDICTIONS_DIR,
        config.FEATURE_CACHE_DIR,
    ]:
        ensure_dir(path)


def to_jsonable(value):
    """Convert common scientific Python values to JSON-compatible objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def save_json(data, path: str | Path) -> Path:
    """Save data as UTF-8 JSON."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(data), file, ensure_ascii=False, indent=2)
    return path


def save_text(text: str, path: str | Path) -> Path:
    """Save text as UTF-8."""
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    return path


def binary_row_to_codes(row: np.ndarray) -> list[str]:
    """Convert one binary label row to IRMAS class codes."""
    return [class_code for index, class_code in enumerate(config.CLASS_CODES) if int(row[index]) == 1]


def class_codes_to_names(class_codes: list[str]) -> list[str]:
    """Convert IRMAS class codes to readable instrument names."""
    return [config.CLASS_NAMES[class_code] for class_code in class_codes]


def load_checkpoint(path: str | Path, device: torch.device):
    """Load a PyTorch checkpoint with compatibility for recent torch versions."""
    path = Path(path)
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def get_state_dict_from_checkpoint(checkpoint):
    """Return a model state dict from a full checkpoint or raw state dict."""
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def get_checkpoint_model_size(checkpoint) -> str:
    """Infer model size from checkpoint metadata, defaulting to baseline small."""
    if not isinstance(checkpoint, dict):
        return "small"
    if "model_size" in checkpoint:
        return str(checkpoint["model_size"])
    train_args = checkpoint.get("train_args")
    if isinstance(train_args, dict) and "model_size" in train_args:
        return str(train_args["model_size"])
    return "small"


def load_model_weights(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device):
    """Load model weights from either a full checkpoint or a raw state dict."""
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(get_state_dict_from_checkpoint(checkpoint))
    return checkpoint
