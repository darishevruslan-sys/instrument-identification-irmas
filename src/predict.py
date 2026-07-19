"""Prediction script for trained IRMAS instrument classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import config
from src.dataset import discover_samples
from src.features import extract_feature
from src.model import build_model
from src.utils import (
    binary_row_to_codes,
    class_codes_to_names,
    ensure_output_dirs,
    get_checkpoint_model_size,
    get_device,
    get_state_dict_from_checkpoint,
    load_checkpoint,
    save_json,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for prediction."""
    parser = argparse.ArgumentParser(description="Predict instruments for wav files.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--audio-path", type=Path, nargs="*", default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--feature-type", choices=config.FEATURE_TYPES, default=config.DEFAULT_FEATURE_TYPE)
    parser.add_argument("--threshold", type=float, default=config.DEFAULT_THRESHOLD)
    parser.add_argument("--max-files", type=int, default=10)
    return parser.parse_args()


def resolve_audio_paths(args: argparse.Namespace) -> list[Path]:
    """Resolve audio files from explicit paths or a data directory."""
    if args.audio_path:
        return list(args.audio_path)
    if args.data_dir:
        samples = discover_samples(args.data_dir, max_files=args.max_files)
        return [sample.path for sample in samples]
    raise SystemExit("Provide --audio-path or --data-dir")


def predict_file(
    model: torch.nn.Module,
    file_path: Path,
    device: torch.device,
    feature_type: str,
    threshold: float,
) -> dict[str, str]:
    """Predict instruments for one wav file."""
    feature = extract_feature(file_path, feature_type=feature_type)
    feature_tensor = torch.from_numpy(feature).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(feature_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]

    binary_pred = (probabilities >= threshold).astype(np.int64)
    predicted_codes = binary_row_to_codes(binary_pred)
    top_indices = np.argsort(probabilities)[::-1][:5]
    top_predictions = [f"{config.CLASS_CODES[index]}:{probabilities[index]:.3f}" for index in top_indices]

    return {
        "path": str(file_path),
        "predicted_codes": ",".join(predicted_codes),
        "predicted_names": ", ".join(class_codes_to_names(predicted_codes)),
        "top_5_probabilities": ", ".join(top_predictions),
    }


def main() -> None:
    """Run predictions and save examples to outputs/predictions."""
    args = parse_args()
    ensure_output_dirs()
    device = get_device()

    checkpoint = load_checkpoint(args.model_path, device)
    model_size = get_checkpoint_model_size(checkpoint)
    model = build_model(num_classes=config.NUM_CLASSES, model_size=model_size).to(device)
    model.load_state_dict(get_state_dict_from_checkpoint(checkpoint))
    print(f"Model size: {model_size}")

    audio_paths = resolve_audio_paths(args)
    rows = [predict_file(model, file_path, device, args.feature_type, args.threshold) for file_path in audio_paths]

    output_csv = config.PREDICTIONS_DIR / f"predict_{args.feature_type}.csv"
    output_json = config.PREDICTIONS_DIR / f"predict_{args.feature_type}.json"
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    save_json(rows, output_json)

    print(f"Predicted files: {len(rows)}")
    print(f"Saved predictions to {output_csv}")


if __name__ == "__main__":
    main()
