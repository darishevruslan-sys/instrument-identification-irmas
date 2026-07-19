"""Visualize Mel-spectrogram and MFCC features for IRMAS audio."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.dataset import discover_samples
from src.features import extract_feature
from src.utils import ensure_dir, ensure_output_dirs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for feature visualization."""
    parser = argparse.ArgumentParser(description="Visualize spectral features for one wav file.")
    parser.add_argument("--data-dir", type=Path, default=config.DEFAULT_DATA_DIR)
    parser.add_argument("--audio-path", type=Path, default=None)
    parser.add_argument("--feature-type", choices=[*config.FEATURE_TYPES, "both"], default=config.DEFAULT_FEATURE_TYPE)
    parser.add_argument("--output-dir", type=Path, default=config.FIGURES_DIR)
    return parser.parse_args()


def resolve_audio_path(data_dir: Path, audio_path: Path | None) -> Path:
    """Use an explicit audio path or the first file discovered in the dataset."""
    if audio_path is not None:
        return audio_path
    samples = discover_samples(data_dir, max_files=1)
    return samples[0].path


def plot_feature(file_path: Path, feature_type: str, output_dir: Path) -> Path:
    """Extract and save one feature image."""
    feature = extract_feature(file_path, feature_type=feature_type)
    matrix = feature[0]

    title = "Mel-spectrogram" if feature_type == "mel" else "MFCC"
    y_label = "Mel bins" if feature_type == "mel" else "MFCC coefficients"
    output_path = output_dir / f"example_{feature_type}.png"

    plt.figure(figsize=(9, 4))
    plt.imshow(matrix, origin="lower", aspect="auto", cmap="magma")
    plt.colorbar(format="%.2f")
    plt.title(f"{title}: {file_path.name}")
    plt.xlabel("Time frames")
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def main() -> None:
    """Create feature visualization images."""
    args = parse_args()
    ensure_output_dirs()
    output_dir = ensure_dir(args.output_dir)
    file_path = resolve_audio_path(args.data_dir, args.audio_path)

    feature_types = list(config.FEATURE_TYPES) if args.feature_type == "both" else [args.feature_type]
    for feature_type in feature_types:
        output_path = plot_feature(file_path, feature_type, output_dir)
        print(f"Saved {feature_type} visualization to {output_path}")


if __name__ == "__main__":
    main()
