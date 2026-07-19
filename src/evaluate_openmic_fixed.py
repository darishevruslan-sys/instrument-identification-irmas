"""Evaluate OpenMIC test metrics with thresholds selected on a separate split."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import config
from src.evaluate import save_dataframe_csv
from src.evaluate_openmic import (
    apply_class_thresholds,
    build_prediction_rows,
    collect_openmic_predictions,
    compute_masked_metrics,
    default_model_path,
    default_thresholds,
    optimize_class_thresholds,
    save_openmic_files,
    threshold_sweep,
)
from src.model import build_model
from src.openmic import OpenMICDataset, discover_openmic_samples, openmic_distribution
from src.utils import (
    ensure_output_dirs,
    get_checkpoint_model_size,
    get_device,
    get_state_dict_from_checkpoint,
    load_checkpoint,
    save_json,
)


def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    """Load an IRMAS checkpoint as an 11-output model."""
    checkpoint = load_checkpoint(model_path, device)
    model_size = get_checkpoint_model_size(checkpoint)
    model = build_model(num_classes=config.NUM_CLASSES, model_size=model_size).to(device)
    model.load_state_dict(get_state_dict_from_checkpoint(checkpoint))
    print(f"Model size: {model_size}")
    if (
        isinstance(checkpoint, dict)
        and "feature_type" in checkpoint
        and checkpoint.get("feature_type") != "mel"
    ):
        warnings.warn(
            f"Checkpoint feature_type={checkpoint.get('feature_type')}; expected mel for the default run",
            stacklevel=2,
        )
    return model


def collect_split_predictions(
    model: torch.nn.Module,
    data_dir: Path,
    split: str,
    feature_type: str,
    batch_size: int,
    num_workers: int,
    positive_threshold: float,
    cache_dir: Path,
    use_cache: bool,
    device: torch.device,
) -> tuple[list, dict[str, np.ndarray | list[str]]]:
    """Load one OpenMIC split and collect model predictions."""
    samples = discover_openmic_samples(
        data_dir=data_dir,
        split=split,
        positive_threshold=positive_threshold,
        require_known_overlap=True,
    )
    dataset = OpenMICDataset(
        samples,
        feature_type=feature_type,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    predictions = collect_openmic_predictions(model, loader, device)
    return samples, predictions


def add_metadata(
    metrics: dict,
    *,
    threshold_source_split: str,
    eval_split: str,
    threshold_mode: str,
    selected_thresholds: dict | float,
    feature_type: str,
    positive_threshold: float,
    samples: list,
) -> dict:
    """Attach common OpenMIC fixed-threshold metadata."""
    metrics.update(
        {
            "dataset": "openmic-2018",
            "threshold_source_split": threshold_source_split,
            "eval_split": eval_split,
            "threshold_mode": threshold_mode,
            "selected_thresholds": selected_thresholds,
            "feature_type": feature_type,
            "num_eval_samples": len(samples),
            "positive_threshold": positive_threshold,
            "class_codes": config.OPENMIC_OVERLAP_CLASS_CODES,
            "class_names": [config.CLASS_NAMES[class_code] for class_code in config.OPENMIC_OVERLAP_CLASS_CODES],
            "positive_distribution": openmic_distribution(samples),
        }
    )
    return metrics


def save_evaluated_predictions(
    prefix: str,
    predictions: dict[str, np.ndarray | list[str]],
    y_pred: np.ndarray,
    metrics: dict,
    report_dict: dict,
    report_text: str,
    tp_fp_fn_rows: list[dict],
    num_examples: int,
) -> None:
    """Save metrics and readable prediction examples for one evaluated split."""
    prediction_rows = build_prediction_rows(
        predictions["sample_keys"],
        predictions["paths"],
        predictions["y_true"],
        y_pred,
        predictions["label_mask"],
        predictions["probabilities"],
        max_rows=num_examples,
    )
    save_openmic_files(
        prefix=prefix,
        metrics=metrics,
        report_dict=report_dict,
        report_text=report_text,
        tp_fp_fn_rows=tp_fp_fn_rows,
        prediction_rows=prediction_rows,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="OpenMIC fixed-threshold train->test evaluation.")
    parser.add_argument("--openmic-dir", type=Path, default=config.DEFAULT_OPENMIC_DIR)
    parser.add_argument("--model-path", type=Path, default=default_model_path())
    parser.add_argument("--feature-type", choices=config.FEATURE_TYPES, default=config.DEFAULT_FEATURE_TYPE)
    parser.add_argument("--threshold-source-split", choices=("train", "test"), default="train")
    parser.add_argument("--eval-split", choices=("train", "test"), default="test")
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--num-examples", type=int, default=30)
    parser.add_argument("--cache-dir", type=Path, default=config.FEATURE_CACHE_DIR)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--prefix", default="openmic_overlap9_fixed")
    return parser.parse_args()


def main() -> None:
    """Select thresholds on one split and evaluate fixed thresholds on another split."""
    args = parse_args()
    ensure_output_dirs()
    device = get_device()
    model = load_model(args.model_path, device)

    threshold_samples, threshold_predictions = collect_split_predictions(
        model=model,
        data_dir=args.openmic_dir,
        split=args.threshold_source_split,
        feature_type=args.feature_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        positive_threshold=args.positive_threshold,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        device=device,
    )
    thresholds = default_thresholds()
    threshold_y_true = threshold_predictions["y_true"]
    threshold_probs = threshold_predictions["probabilities"]
    threshold_mask = threshold_predictions["label_mask"]

    sweep_rows = threshold_sweep(threshold_y_true, threshold_probs, threshold_mask, thresholds)
    best_common = max(sweep_rows, key=lambda row: row["macro_f1"])
    best_common_threshold = float(best_common["threshold"])
    class_thresholds, class_threshold_rows = optimize_class_thresholds(
        threshold_y_true,
        threshold_probs,
        threshold_mask,
        thresholds,
    )

    save_dataframe_csv(
        pd.DataFrame(sweep_rows),
        config.METRICS_DIR / f"{args.prefix}_source_common_threshold_sweep.csv",
    )
    save_json(sweep_rows, config.METRICS_DIR / f"{args.prefix}_source_common_threshold_sweep.json")
    save_dataframe_csv(
        pd.DataFrame(class_threshold_rows),
        config.METRICS_DIR / f"{args.prefix}_source_class_thresholds.csv",
    )
    save_json(class_threshold_rows, config.METRICS_DIR / f"{args.prefix}_source_class_thresholds.json")
    save_json(
        {
            "threshold_source_split": args.threshold_source_split,
            "num_threshold_samples": len(threshold_samples),
            "best_common_threshold": best_common_threshold,
            "best_common_source_metrics": best_common,
            "class_thresholds": class_thresholds,
        },
        config.METRICS_DIR / f"{args.prefix}_selected_thresholds.json",
    )

    eval_samples, eval_predictions = collect_split_predictions(
        model=model,
        data_dir=args.openmic_dir,
        split=args.eval_split,
        feature_type=args.feature_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        positive_threshold=args.positive_threshold,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        device=device,
    )
    eval_y_true = eval_predictions["y_true"]
    eval_probs = eval_predictions["probabilities"]
    eval_mask = eval_predictions["label_mask"]

    common_pred = (eval_probs >= best_common_threshold).astype(np.int64)
    common_metrics, common_report_dict, common_report_text, common_tp_fp_fn_rows = compute_masked_metrics(
        y_true=eval_y_true,
        y_pred=common_pred,
        label_mask=eval_mask,
        probabilities=eval_probs,
        threshold=best_common_threshold,
    )
    add_metadata(
        common_metrics,
        threshold_source_split=args.threshold_source_split,
        eval_split=args.eval_split,
        threshold_mode="common",
        selected_thresholds=best_common_threshold,
        feature_type=args.feature_type,
        positive_threshold=args.positive_threshold,
        samples=eval_samples,
    )
    save_evaluated_predictions(
        prefix=f"{args.prefix}_common",
        predictions=eval_predictions,
        y_pred=common_pred,
        metrics=common_metrics,
        report_dict=common_report_dict,
        report_text=common_report_text,
        tp_fp_fn_rows=common_tp_fp_fn_rows,
        num_examples=args.num_examples,
    )

    class_pred = apply_class_thresholds(eval_probs, class_thresholds)
    class_metrics, class_report_dict, class_report_text, class_tp_fp_fn_rows = compute_masked_metrics(
        y_true=eval_y_true,
        y_pred=class_pred,
        label_mask=eval_mask,
        probabilities=eval_probs,
        threshold=0.0,
    )
    add_metadata(
        class_metrics,
        threshold_source_split=args.threshold_source_split,
        eval_split=args.eval_split,
        threshold_mode="per_class",
        selected_thresholds=class_thresholds,
        feature_type=args.feature_type,
        positive_threshold=args.positive_threshold,
        samples=eval_samples,
    )
    save_evaluated_predictions(
        prefix=f"{args.prefix}_class_thresholds",
        predictions=eval_predictions,
        y_pred=class_pred,
        metrics=class_metrics,
        report_dict=class_report_dict,
        report_text=class_report_text,
        tp_fp_fn_rows=class_tp_fp_fn_rows,
        num_examples=args.num_examples,
    )

    print(f"Threshold source split: {args.threshold_source_split} ({len(threshold_samples)} samples)")
    print(f"Eval split: {args.eval_split} ({len(eval_samples)} samples)")
    print(
        "Fixed common threshold: "
        f"{best_common_threshold:.2f} | "
        f"test macro_f1={common_metrics['macro_f1']:.4f} | "
        f"micro_f1={common_metrics['micro_f1']:.4f}"
    )
    print(
        "Fixed class thresholds: "
        f"test macro_f1={class_metrics['macro_f1']:.4f} | "
        f"micro_f1={class_metrics['micro_f1']:.4f} | "
        f"precision={class_metrics['precision']:.4f} | "
        f"recall={class_metrics['recall']:.4f}"
    )


if __name__ == "__main__":
    main()
