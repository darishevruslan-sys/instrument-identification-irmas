"""Evaluation script and metric helpers for multi-label classification."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from src import config
from src.dataset import IRMASDataset, discover_samples, split_samples
from src.model import build_model
from src.utils import (
    binary_row_to_codes,
    class_codes_to_names,
    ensure_output_dirs,
    get_checkpoint_model_size,
    get_state_dict_from_checkpoint,
    load_checkpoint,
    save_json,
    save_text,
    get_device,
)


def collect_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray | list[str]]:
    """Collect true labels, predicted labels and probabilities from a dataloader."""
    model.eval()
    all_true: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_paths: list[str] = []
    all_class_codes: list[str] = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            labels = batch["labels"].cpu().numpy()

            logits = model(features)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_true.append(labels.astype(np.int64))
            all_probs.append(probs)
            all_paths.extend(list(batch["path"]))
            all_class_codes.extend(list(batch["class_code"]))

    return {
        "y_true": np.concatenate(all_true, axis=0),
        "probabilities": np.concatenate(all_probs, axis=0),
        "paths": all_paths,
        "class_codes": all_class_codes,
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None = None,
    threshold: float = config.DEFAULT_THRESHOLD,
) -> tuple[dict[str, float], dict, str, list[dict[str, int | str]]]:
    """Compute multi-label metrics and per-class TP/FP/FN table."""
    target_names = [config.CLASS_NAMES[class_code] for class_code in config.CLASS_CODES]
    predicted_label_count = int(y_pred.sum())
    avg_predicted_labels = float(predicted_label_count / max(len(y_pred), 1))
    metrics = {
        "threshold": float(threshold),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "predicted_label_count": predicted_label_count,
        "avg_predicted_labels_per_sample": avg_predicted_labels,
    }
    if probabilities is not None:
        top1_pred = np.zeros_like(y_true, dtype=np.int64)
        top1_indices = np.argmax(probabilities, axis=1)
        top1_pred[np.arange(len(top1_indices)), top1_indices] = 1
        metrics["top1_accuracy"] = float(accuracy_score(y_true, top1_pred))
        metrics["top1_macro_f1"] = float(f1_score(y_true, top1_pred, average="macro", zero_division=0))

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
    )

    tp_fp_fn_rows: list[dict[str, int | str]] = []
    for index, class_code in enumerate(config.CLASS_CODES):
        true_col = y_true[:, index]
        pred_col = y_pred[:, index]
        tp = int(((true_col == 1) & (pred_col == 1)).sum())
        fp = int(((true_col == 0) & (pred_col == 1)).sum())
        fn = int(((true_col == 1) & (pred_col == 0)).sum())
        tn = int(((true_col == 0) & (pred_col == 0)).sum())
        tp_fp_fn_rows.append(
            {
                "class_code": class_code,
                "class_name": config.CLASS_NAMES[class_code],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "support": int(true_col.sum()),
            }
        )

    return metrics, report_dict, report_text, tp_fp_fn_rows


def threshold_sweep(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, float]]:
    """Evaluate metrics for multiple thresholds using cached probabilities."""
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        y_pred = (probabilities >= threshold).astype(np.int64)
        metrics, _, _, _ = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
            threshold=threshold,
        )
        rows.append(metrics)
    return rows


def optimize_class_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: list[float],
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    """Find the best threshold for each class by binary F1-score."""
    best_thresholds: dict[str, float] = {}
    rows: list[dict[str, float | int | str]] = []

    for index, class_code in enumerate(config.CLASS_CODES):
        true_col = y_true[:, index]
        prob_col = probabilities[:, index]
        best_row: dict[str, float | int | str] | None = None

        for threshold in thresholds:
            pred_col = (prob_col >= threshold).astype(np.int64)
            f1 = float(f1_score(true_col, pred_col, zero_division=0))
            precision = float(precision_score(true_col, pred_col, zero_division=0))
            recall = float(recall_score(true_col, pred_col, zero_division=0))
            row = {
                "class_code": class_code,
                "class_name": config.CLASS_NAMES[class_code],
                "threshold": threshold,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "support": int(true_col.sum()),
                "predicted_count": int(pred_col.sum()),
            }
            if best_row is None or f1 > float(best_row["f1"]):
                best_row = row

        assert best_row is not None
        best_thresholds[class_code] = float(best_row["threshold"])
        rows.append(best_row)

    return best_thresholds, rows


def apply_class_thresholds(probabilities: np.ndarray, class_thresholds: dict[str, float]) -> np.ndarray:
    """Convert probabilities to binary predictions using class-specific thresholds."""
    thresholds = np.array([class_thresholds[class_code] for class_code in config.CLASS_CODES], dtype=np.float32)
    return (probabilities >= thresholds.reshape(1, -1)).astype(np.int64)


def default_thresholds() -> list[float]:
    """Return thresholds from 0.10 to 0.90 inclusive."""
    return [round(value, 2) for value in np.arange(0.10, 0.91, 0.05)]


def build_prediction_rows(
    paths: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    max_rows: int | None = None,
) -> list[dict[str, str | float]]:
    """Build readable prediction examples for saving to CSV/JSON."""
    rows: list[dict[str, str | float]] = []
    row_count = len(paths) if max_rows is None else min(len(paths), max_rows)

    for row_index in range(row_count):
        true_codes = binary_row_to_codes(y_true[row_index])
        predicted_codes = binary_row_to_codes(y_pred[row_index])
        prob_row = probabilities[row_index]
        top_indices = np.argsort(prob_row)[::-1][:3]
        top_predictions = [
            f"{config.CLASS_CODES[index]}:{prob_row[index]:.3f}" for index in top_indices
        ]
        rows.append(
            {
                "path": paths[row_index],
                "true_codes": ",".join(true_codes),
                "true_names": ", ".join(class_codes_to_names(true_codes)),
                "predicted_codes": ",".join(predicted_codes),
                "predicted_names": ", ".join(class_codes_to_names(predicted_codes)),
                "top_3_probabilities": ", ".join(top_predictions),
            }
        )

    return rows


def save_evaluation_files(
    prefix: str,
    metrics: dict,
    report_dict: dict,
    report_text: str,
    tp_fp_fn_rows: list[dict],
    prediction_rows: list[dict],
) -> None:
    """Save metrics, reports, TP/FP/FN table and prediction examples."""
    save_json(metrics, config.METRICS_DIR / f"{prefix}_metrics.json")
    save_json(report_dict, config.METRICS_DIR / f"{prefix}_classification_report.json")
    save_text(report_text, config.METRICS_DIR / f"{prefix}_classification_report.txt")
    save_dataframe_csv(pd.DataFrame(tp_fp_fn_rows), config.METRICS_DIR / f"{prefix}_tp_fp_fn.csv")
    save_json(tp_fp_fn_rows, config.METRICS_DIR / f"{prefix}_tp_fp_fn.json")
    save_dataframe_csv(pd.DataFrame(prediction_rows), config.PREDICTIONS_DIR / f"{prefix}_predictions.csv")
    save_json(prediction_rows, config.PREDICTIONS_DIR / f"{prefix}_predictions.json")


def save_dataframe_csv(dataframe: pd.DataFrame, path: Path) -> Path:
    """Save a CSV file, using a timestamped fallback if the target is locked."""
    try:
        dataframe.to_csv(path, index=False)
        return path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        dataframe.to_csv(fallback_path, index=False)
        warnings.warn(
            f"Could not write {path} because it is locked. Saved {fallback_path} instead.",
            stacklevel=2,
        )
        return fallback_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate an IRMAS instrument classifier.")
    parser.add_argument("--data-dir", type=Path, default=config.DEFAULT_DATA_DIR)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--feature-type", choices=config.FEATURE_TYPES, default=config.DEFAULT_FEATURE_TYPE)
    parser.add_argument("--threshold", type=float, default=config.DEFAULT_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--validation-size", type=float, default=config.VALIDATION_SIZE)
    parser.add_argument("--random-state", type=int, default=config.RANDOM_STATE)
    parser.add_argument("--use-all-data", action="store_true")
    parser.add_argument("--num-examples", type=int, default=30)
    parser.add_argument("--sweep-thresholds", action="store_true")
    parser.add_argument("--optimize-class-thresholds", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=config.FEATURE_CACHE_DIR)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run model evaluation on the validation split or all discovered data."""
    args = parse_args()
    ensure_output_dirs()
    device = get_device()

    samples = discover_samples(args.data_dir, max_files=args.max_files)
    if args.use_all_data:
        eval_samples = samples
    else:
        _, eval_samples = split_samples(
            samples,
            validation_size=args.validation_size,
            random_state=args.random_state,
        )

    eval_dataset = IRMASDataset(
        eval_samples,
        feature_type=args.feature_type,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    checkpoint = load_checkpoint(args.model_path, device)
    model_size = get_checkpoint_model_size(checkpoint)
    model = build_model(num_classes=config.NUM_CLASSES, model_size=model_size).to(device)
    model.load_state_dict(get_state_dict_from_checkpoint(checkpoint))
    print(f"Model size: {model_size}")
    if (
        isinstance(checkpoint, dict)
        and "feature_type" in checkpoint
        and checkpoint.get("feature_type") != args.feature_type
    ):
        warnings.warn(
            f"Checkpoint feature_type={checkpoint.get('feature_type')} but evaluation uses {args.feature_type}",
            stacklevel=2,
        )

    predictions = collect_predictions(model, eval_loader, device)
    probabilities = predictions["probabilities"]
    y_true = predictions["y_true"]
    y_pred = (probabilities >= args.threshold).astype(np.int64)
    metrics, report_dict, report_text, tp_fp_fn_rows = compute_metrics(
        y_true,
        y_pred,
        probabilities=probabilities,
        threshold=args.threshold,
    )
    metrics["num_eval_samples"] = len(eval_samples)
    metrics["feature_type"] = args.feature_type

    prediction_rows = build_prediction_rows(
        predictions["paths"],
        y_true,
        y_pred,
        probabilities,
        max_rows=args.num_examples,
    )
    save_evaluation_files(
        prefix=f"evaluation_{args.feature_type}",
        metrics=metrics,
        report_dict=report_dict,
        report_text=report_text,
        tp_fp_fn_rows=tp_fp_fn_rows,
        prediction_rows=prediction_rows,
    )

    if args.sweep_thresholds:
        sweep_rows = threshold_sweep(y_true, probabilities, default_thresholds())
        sweep_path = config.METRICS_DIR / f"evaluation_{args.feature_type}_threshold_sweep.csv"
        pd.DataFrame(sweep_rows).to_csv(sweep_path, index=False)
        save_json(sweep_rows, config.METRICS_DIR / f"evaluation_{args.feature_type}_threshold_sweep.json")
        best_row = max(sweep_rows, key=lambda row: row["macro_f1"])
        print(
            "Best threshold by macro_f1: "
            f"{best_row['threshold']:.2f} | "
            f"macro_f1={best_row['macro_f1']:.4f} | "
            f"micro_f1={best_row['micro_f1']:.4f} | "
            f"precision={best_row['precision']:.4f} | "
            f"recall={best_row['recall']:.4f} | "
            f"avg_labels={best_row['avg_predicted_labels_per_sample']:.2f}"
        )
        print(f"Saved threshold sweep to {sweep_path}")

    if args.optimize_class_thresholds:
        class_thresholds, class_threshold_rows = optimize_class_thresholds(
            y_true,
            probabilities,
            default_thresholds(),
        )
        class_y_pred = apply_class_thresholds(probabilities, class_thresholds)
        class_metrics, class_report_dict, class_report_text, class_tp_fp_fn_rows = compute_metrics(
            y_true,
            class_y_pred,
            probabilities=probabilities,
            threshold=0.0,
        )
        class_metrics.update(
            {
                "threshold_mode": "per_class",
                "class_thresholds": class_thresholds,
                "num_eval_samples": len(eval_samples),
                "feature_type": args.feature_type,
            }
        )
        class_prediction_rows = build_prediction_rows(
            predictions["paths"],
            y_true,
            class_y_pred,
            probabilities,
            max_rows=args.num_examples,
        )
        prefix = f"evaluation_{args.feature_type}_class_thresholds"
        save_evaluation_files(
            prefix=prefix,
            metrics=class_metrics,
            report_dict=class_report_dict,
            report_text=class_report_text,
            tp_fp_fn_rows=class_tp_fp_fn_rows,
            prediction_rows=class_prediction_rows,
        )
        pd.DataFrame(class_threshold_rows).to_csv(
            config.METRICS_DIR / f"{prefix}_thresholds.csv",
            index=False,
        )
        save_json(class_threshold_rows, config.METRICS_DIR / f"{prefix}_thresholds.json")
        print(
            "Class thresholds macro_f1: "
            f"{class_metrics['macro_f1']:.4f} | "
            f"micro_f1={class_metrics['micro_f1']:.4f} | "
            f"precision={class_metrics['precision']:.4f} | "
            f"recall={class_metrics['recall']:.4f} | "
            f"avg_labels={class_metrics['avg_predicted_labels_per_sample']:.2f}"
        )
        print(f"Saved class-threshold evaluation to outputs/metrics/{prefix}_metrics.json")

    print(f"Evaluation samples: {len(eval_samples)}")
    print(f"Device: {device}")
    print(f"micro_f1: {metrics['micro_f1']:.4f}")
    print(f"macro_f1: {metrics['macro_f1']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall: {metrics['recall']:.4f}")
    print(f"top1_accuracy: {metrics['top1_accuracy']:.4f}")
    print(f"avg predicted labels/sample: {metrics['avg_predicted_labels_per_sample']:.2f}")
    print(f"subset_accuracy: {metrics['subset_accuracy']:.4f}")


if __name__ == "__main__":
    main()
