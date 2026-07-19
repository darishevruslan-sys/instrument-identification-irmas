"""Evaluate an IRMAS checkpoint on the OpenMIC 2018 overlap-9 labels."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from src import config
from src.evaluate import save_dataframe_csv
from src.model import build_model
from src.openmic import OpenMICDataset, discover_openmic_samples, openmic_distribution
from src.utils import (
    ensure_output_dirs,
    get_checkpoint_model_size,
    get_device,
    get_state_dict_from_checkpoint,
    load_checkpoint,
    save_json,
    save_text,
)


def default_model_path() -> Path:
    """Return the preferred IRMAS checkpoint for OpenMIC evaluation."""
    preferred = config.MODELS_DIR / "best_model_mel_v2_best.pth"
    if preferred.exists():
        return preferred
    return config.MODELS_DIR / "best_model_mel_v2.pth"


def collect_openmic_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray | list[str]]:
    """Collect overlap-9 labels, masks and probabilities from a dataloader."""
    model.eval()
    all_true: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_paths: list[str] = []
    all_sample_keys: list[str] = []
    overlap_indices = np.array(config.OPENMIC_OVERLAP_CLASS_TO_IRMAS_INDEX, dtype=np.int64)

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            logits = model(features)
            probabilities = torch.sigmoid(logits).cpu().numpy()

            all_true.append(batch["labels"].cpu().numpy().astype(np.int64))
            all_masks.append(batch["label_mask"].cpu().numpy().astype(bool))
            all_probs.append(probabilities[:, overlap_indices])
            all_paths.extend(list(batch["path"]))
            all_sample_keys.extend(list(batch["sample_key"]))

    return {
        "y_true": np.concatenate(all_true, axis=0),
        "label_mask": np.concatenate(all_masks, axis=0),
        "probabilities": np.concatenate(all_probs, axis=0),
        "paths": all_paths,
        "sample_keys": all_sample_keys,
    }


def _binary_metric(metric_fn, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute a binary metric with zero-division handling."""
    return float(metric_fn(y_true, y_pred, zero_division=0))


def _masked_subset_accuracy(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """Return exact-match accuracy over known labels only."""
    valid_rows = mask.any(axis=1)
    if not bool(valid_rows.any()):
        return 0.0
    known_equal = (y_true == y_pred) | ~mask
    return float(known_equal[valid_rows].all(axis=1).mean())


def _top1_positive_accuracy(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Return whether the top probability is one of the known positive labels."""
    positive_rows = y_true.sum(axis=1) > 0
    if not bool(positive_rows.any()):
        return 0.0
    top_indices = np.argmax(probabilities[positive_rows], axis=1)
    true_rows = y_true[positive_rows]
    return float(true_rows[np.arange(len(top_indices)), top_indices].mean())


def compute_masked_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_mask: np.ndarray,
    probabilities: np.ndarray | None = None,
    threshold: float = config.DEFAULT_THRESHOLD,
    include_groups: bool = True,
) -> tuple[dict[str, float | int], dict, str, list[dict[str, float | int | str]]]:
    """Compute masked multi-label metrics for partially annotated OpenMIC labels."""
    mask = label_mask.astype(bool)
    if y_true.shape != y_pred.shape or y_true.shape != mask.shape:
        raise ValueError("y_true, y_pred and label_mask must have the same shape")
    if not bool(mask.any()):
        raise ValueError("No known OpenMIC labels available for evaluation")

    flat_true = y_true[mask]
    flat_pred = y_pred[mask]

    tp_fp_fn_rows: list[dict[str, float | int | str]] = []
    class_precisions: list[float] = []
    class_recalls: list[float] = []
    class_f1s: list[float] = []

    for index, class_code in enumerate(config.OPENMIC_OVERLAP_CLASS_CODES):
        class_mask = mask[:, index]
        true_col = y_true[class_mask, index]
        pred_col = y_pred[class_mask, index]
        tp = int(((true_col == 1) & (pred_col == 1)).sum())
        fp = int(((true_col == 0) & (pred_col == 1)).sum())
        fn = int(((true_col == 1) & (pred_col == 0)).sum())
        tn = int(((true_col == 0) & (pred_col == 0)).sum())
        if bool(class_mask.any()):
            precision = _binary_metric(precision_score, true_col, pred_col)
            recall = _binary_metric(recall_score, true_col, pred_col)
            f1 = _binary_metric(f1_score, true_col, pred_col)
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        class_precisions.append(precision)
        class_recalls.append(recall)
        class_f1s.append(f1)
        tp_fp_fn_rows.append(
            {
                "class_code": class_code,
                "class_name": config.CLASS_NAMES[class_code],
                "known_count": int(class_mask.sum()),
                "support": int(true_col.sum()),
                "predicted_count_known": int(pred_col.sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

    known_predicted_count = int(y_pred[mask].sum())
    positive_counts = y_true.sum(axis=1)
    metrics: dict[str, float | int] = {
        "threshold": float(threshold),
        "micro_f1": _binary_metric(f1_score, flat_true, flat_pred),
        "macro_f1": float(np.mean(class_f1s)),
        "precision": _binary_metric(precision_score, flat_true, flat_pred),
        "recall": _binary_metric(recall_score, flat_true, flat_pred),
        "macro_precision": float(np.mean(class_precisions)),
        "macro_recall": float(np.mean(class_recalls)),
        "masked_subset_accuracy": _masked_subset_accuracy(y_true, y_pred, mask),
        "known_label_count": int(mask.sum()),
        "positive_known_label_count": int(y_true[mask].sum()),
        "predicted_label_count_known": known_predicted_count,
        "avg_predicted_labels_per_sample": float(y_pred.sum(axis=1).mean()),
        "avg_known_predicted_labels_per_sample": float(known_predicted_count / len(y_pred)),
        "num_positive_samples": int((positive_counts > 0).sum()),
        "num_multi_positive_samples": int((positive_counts >= 2).sum()),
    }
    if probabilities is not None:
        metrics["top1_positive_accuracy"] = _top1_positive_accuracy(y_true, probabilities)

    if include_groups:
        multi_rows = positive_counts >= 2
        if bool(multi_rows.any()):
            multi_metrics, _, _, _ = compute_masked_metrics(
                y_true[multi_rows],
                y_pred[multi_rows],
                mask[multi_rows],
                None if probabilities is None else probabilities[multi_rows],
                threshold=threshold,
                include_groups=False,
            )
            for key in ["micro_f1", "macro_f1", "precision", "recall", "masked_subset_accuracy"]:
                metrics[f"multi_positive_{key}"] = float(multi_metrics[key])

    report_dict: dict[str, dict | float] = {
        row["class_name"]: row for row in tp_fp_fn_rows
    }
    report_dict["micro_avg"] = {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["micro_f1"],
        "support": metrics["positive_known_label_count"],
    }
    report_dict["macro_avg"] = {
        "precision": metrics["macro_precision"],
        "recall": metrics["macro_recall"],
        "f1": metrics["macro_f1"],
        "support": metrics["positive_known_label_count"],
    }

    report_text = build_report_text(tp_fp_fn_rows, metrics)
    return metrics, report_dict, report_text, tp_fp_fn_rows


def build_report_text(
    tp_fp_fn_rows: list[dict[str, float | int | str]],
    metrics: dict[str, float | int],
) -> str:
    """Build a compact text classification report."""
    lines = [
        "OpenMIC overlap-9 masked classification report",
        "",
        f"{'class':<18} {'known':>7} {'support':>7} {'prec':>8} {'recall':>8} {'f1':>8}",
    ]
    for row in tp_fp_fn_rows:
        lines.append(
            f"{str(row['class_name']):<18} "
            f"{int(row['known_count']):>7} "
            f"{int(row['support']):>7} "
            f"{float(row['precision']):>8.4f} "
            f"{float(row['recall']):>8.4f} "
            f"{float(row['f1']):>8.4f}"
        )
    lines.extend(
        [
            "",
            f"micro_f1: {float(metrics['micro_f1']):.4f}",
            f"macro_f1: {float(metrics['macro_f1']):.4f}",
            f"precision: {float(metrics['precision']):.4f}",
            f"recall: {float(metrics['recall']):.4f}",
            f"masked_subset_accuracy: {float(metrics['masked_subset_accuracy']):.4f}",
        ]
    )
    return "\n".join(lines)


def threshold_sweep(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    label_mask: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, float | int]]:
    """Evaluate masked metrics for multiple common thresholds."""
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        y_pred = (probabilities >= threshold).astype(np.int64)
        metrics, _, _, _ = compute_masked_metrics(
            y_true=y_true,
            y_pred=y_pred,
            label_mask=label_mask,
            probabilities=probabilities,
            threshold=threshold,
        )
        rows.append(metrics)
    return rows


def optimize_class_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    label_mask: np.ndarray,
    thresholds: list[float],
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    """Find the best threshold for each overlap class by masked binary F1."""
    best_thresholds: dict[str, float] = {}
    rows: list[dict[str, float | int | str]] = []
    mask = label_mask.astype(bool)

    for index, class_code in enumerate(config.OPENMIC_OVERLAP_CLASS_CODES):
        class_mask = mask[:, index]
        true_col = y_true[class_mask, index]
        prob_col = probabilities[class_mask, index]
        best_row: dict[str, float | int | str] | None = None

        for threshold in thresholds:
            pred_col = (prob_col >= threshold).astype(np.int64)
            if bool(class_mask.any()):
                f1 = _binary_metric(f1_score, true_col, pred_col)
                precision = _binary_metric(precision_score, true_col, pred_col)
                recall = _binary_metric(recall_score, true_col, pred_col)
            else:
                f1 = 0.0
                precision = 0.0
                recall = 0.0
            row = {
                "class_code": class_code,
                "class_name": config.CLASS_NAMES[class_code],
                "threshold": threshold,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "known_count": int(class_mask.sum()),
                "support": int(true_col.sum()),
                "predicted_count_known": int(pred_col.sum()),
            }
            if best_row is None or f1 > float(best_row["f1"]):
                best_row = row

        assert best_row is not None
        best_thresholds[class_code] = float(best_row["threshold"])
        rows.append(best_row)

    return best_thresholds, rows


def apply_class_thresholds(probabilities: np.ndarray, class_thresholds: dict[str, float]) -> np.ndarray:
    """Convert probabilities to binary predictions using overlap-class thresholds."""
    thresholds = np.array(
        [class_thresholds[class_code] for class_code in config.OPENMIC_OVERLAP_CLASS_CODES],
        dtype=np.float32,
    )
    return (probabilities >= thresholds.reshape(1, -1)).astype(np.int64)


def default_thresholds() -> list[float]:
    """Return thresholds from 0.10 to 0.90 inclusive."""
    return [round(value, 2) for value in np.arange(0.10, 0.91, 0.05)]


def _row_codes(binary_row: np.ndarray) -> list[str]:
    """Convert one overlap-9 binary row to IRMAS class codes."""
    return [
        class_code
        for index, class_code in enumerate(config.OPENMIC_OVERLAP_CLASS_CODES)
        if int(binary_row[index]) == 1
    ]


def _codes_to_names(class_codes: list[str]) -> list[str]:
    """Convert IRMAS class codes to readable instrument names."""
    return [config.CLASS_NAMES[class_code] for class_code in class_codes]


def build_prediction_rows(
    sample_keys: list[str],
    paths: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_mask: np.ndarray,
    probabilities: np.ndarray,
    max_rows: int | None = None,
) -> list[dict[str, str | float | int]]:
    """Build readable OpenMIC prediction examples for saving."""
    rows: list[dict[str, str | float | int]] = []
    row_count = len(paths) if max_rows is None else min(len(paths), max_rows)

    for row_index in range(row_count):
        true_codes = _row_codes(y_true[row_index])
        known_codes = _row_codes(label_mask[row_index].astype(np.int64))
        predicted_codes = _row_codes(y_pred[row_index])
        predicted_known_codes = _row_codes((y_pred[row_index] & label_mask[row_index]).astype(np.int64))
        prob_row = probabilities[row_index]
        top_indices = np.argsort(prob_row)[::-1][:3]
        top_predictions = [
            f"{config.OPENMIC_OVERLAP_CLASS_CODES[index]}:{prob_row[index]:.3f}"
            for index in top_indices
        ]
        rows.append(
            {
                "sample_key": sample_keys[row_index],
                "path": paths[row_index],
                "known_codes": ",".join(known_codes),
                "known_names": ", ".join(_codes_to_names(known_codes)),
                "true_codes": ",".join(true_codes),
                "true_names": ", ".join(_codes_to_names(true_codes)),
                "predicted_codes": ",".join(predicted_codes),
                "predicted_names": ", ".join(_codes_to_names(predicted_codes)),
                "predicted_known_codes": ",".join(predicted_known_codes),
                "top_3_probabilities": ", ".join(top_predictions),
                "positive_label_count": int(y_true[row_index].sum()),
                "known_label_count": int(label_mask[row_index].sum()),
            }
        )

    return rows


def save_openmic_files(
    prefix: str,
    metrics: dict,
    report_dict: dict,
    report_text: str,
    tp_fp_fn_rows: list[dict],
    prediction_rows: list[dict],
) -> None:
    """Save OpenMIC metrics, reports, TP/FP/FN table and prediction examples."""
    save_json(metrics, config.METRICS_DIR / f"{prefix}_metrics.json")
    save_json(report_dict, config.METRICS_DIR / f"{prefix}_classification_report.json")
    save_text(report_text, config.METRICS_DIR / f"{prefix}_classification_report.txt")
    save_dataframe_csv(pd.DataFrame(tp_fp_fn_rows), config.METRICS_DIR / f"{prefix}_tp_fp_fn.csv")
    save_json(tp_fp_fn_rows, config.METRICS_DIR / f"{prefix}_tp_fp_fn.json")
    save_dataframe_csv(pd.DataFrame(prediction_rows), config.PREDICTIONS_DIR / f"{prefix}_predictions.csv")
    save_json(prediction_rows, config.PREDICTIONS_DIR / f"{prefix}_predictions.json")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for OpenMIC evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate an IRMAS checkpoint on OpenMIC overlap-9.")
    parser.add_argument("--openmic-dir", type=Path, default=config.DEFAULT_OPENMIC_DIR)
    parser.add_argument("--model-path", type=Path, default=default_model_path())
    parser.add_argument("--feature-type", choices=config.FEATURE_TYPES, default=config.DEFAULT_FEATURE_TYPE)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--num-examples", type=int, default=30)
    parser.add_argument("--sweep-thresholds", action="store_true")
    parser.add_argument("--optimize-class-thresholds", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=config.FEATURE_CACHE_DIR)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--prefix", default="openmic_overlap9")
    return parser.parse_args()


def main() -> None:
    """Run OpenMIC overlap-9 evaluation."""
    args = parse_args()
    ensure_output_dirs()
    device = get_device()

    samples = discover_openmic_samples(
        data_dir=args.openmic_dir,
        split=args.split,
        max_files=args.max_files,
        positive_threshold=args.positive_threshold,
        require_known_overlap=True,
    )
    eval_dataset = OpenMICDataset(
        samples,
        feature_type=args.feature_type,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
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
            f"Checkpoint feature_type={checkpoint.get('feature_type')} "
            f"but evaluation uses {args.feature_type}",
            stacklevel=2,
        )

    predictions = collect_openmic_predictions(model, eval_loader, device)
    probabilities = predictions["probabilities"]
    y_true = predictions["y_true"]
    label_mask = predictions["label_mask"]
    y_pred = (probabilities >= args.threshold).astype(np.int64)

    metrics, report_dict, report_text, tp_fp_fn_rows = compute_masked_metrics(
        y_true=y_true,
        y_pred=y_pred,
        label_mask=label_mask,
        probabilities=probabilities,
        threshold=args.threshold,
    )
    metrics.update(
        {
            "dataset": "openmic-2018",
            "split": args.split,
            "feature_type": args.feature_type,
            "num_eval_samples": len(samples),
            "positive_threshold": args.positive_threshold,
            "class_codes": config.OPENMIC_OVERLAP_CLASS_CODES,
            "class_names": [config.CLASS_NAMES[class_code] for class_code in config.OPENMIC_OVERLAP_CLASS_CODES],
            "positive_distribution": openmic_distribution(samples),
        }
    )

    prediction_rows = build_prediction_rows(
        predictions["sample_keys"],
        predictions["paths"],
        y_true,
        y_pred,
        label_mask,
        probabilities,
        max_rows=args.num_examples,
    )
    save_openmic_files(
        prefix=args.prefix,
        metrics=metrics,
        report_dict=report_dict,
        report_text=report_text,
        tp_fp_fn_rows=tp_fp_fn_rows,
        prediction_rows=prediction_rows,
    )

    if args.sweep_thresholds:
        sweep_rows = threshold_sweep(y_true, probabilities, label_mask, default_thresholds())
        sweep_path = config.METRICS_DIR / f"{args.prefix}_threshold_sweep.csv"
        save_dataframe_csv(pd.DataFrame(sweep_rows), sweep_path)
        save_json(sweep_rows, config.METRICS_DIR / f"{args.prefix}_threshold_sweep.json")
        best_row = max(sweep_rows, key=lambda row: row["macro_f1"])
        print(
            "Best OpenMIC threshold by macro_f1: "
            f"{best_row['threshold']:.2f} | "
            f"macro_f1={best_row['macro_f1']:.4f} | "
            f"micro_f1={best_row['micro_f1']:.4f} | "
            f"precision={best_row['precision']:.4f} | "
            f"recall={best_row['recall']:.4f}"
        )
        print(f"Saved threshold sweep to {sweep_path}")

    if args.optimize_class_thresholds:
        class_thresholds, class_threshold_rows = optimize_class_thresholds(
            y_true,
            probabilities,
            label_mask,
            default_thresholds(),
        )
        class_y_pred = apply_class_thresholds(probabilities, class_thresholds)
        class_metrics, class_report_dict, class_report_text, class_tp_fp_fn_rows = compute_masked_metrics(
            y_true=y_true,
            y_pred=class_y_pred,
            label_mask=label_mask,
            probabilities=probabilities,
            threshold=0.0,
        )
        class_metrics.update(
            {
                "dataset": "openmic-2018",
                "split": args.split,
                "feature_type": args.feature_type,
                "num_eval_samples": len(samples),
                "positive_threshold": args.positive_threshold,
                "threshold_mode": "per_class",
                "class_thresholds": class_thresholds,
                "class_codes": config.OPENMIC_OVERLAP_CLASS_CODES,
                "class_names": [config.CLASS_NAMES[class_code] for class_code in config.OPENMIC_OVERLAP_CLASS_CODES],
                "positive_distribution": openmic_distribution(samples),
            }
        )
        class_prediction_rows = build_prediction_rows(
            predictions["sample_keys"],
            predictions["paths"],
            y_true,
            class_y_pred,
            label_mask,
            probabilities,
            max_rows=args.num_examples,
        )
        class_prefix = f"{args.prefix}_class_thresholds"
        save_openmic_files(
            prefix=class_prefix,
            metrics=class_metrics,
            report_dict=class_report_dict,
            report_text=class_report_text,
            tp_fp_fn_rows=class_tp_fp_fn_rows,
            prediction_rows=class_prediction_rows,
        )
        save_dataframe_csv(
            pd.DataFrame(class_threshold_rows),
            config.METRICS_DIR / f"{class_prefix}_thresholds.csv",
        )
        save_json(class_threshold_rows, config.METRICS_DIR / f"{class_prefix}_thresholds.json")
        print(
            "OpenMIC class thresholds macro_f1: "
            f"{class_metrics['macro_f1']:.4f} | "
            f"micro_f1={class_metrics['micro_f1']:.4f} | "
            f"precision={class_metrics['precision']:.4f} | "
            f"recall={class_metrics['recall']:.4f}"
        )
        print(f"Saved class-threshold evaluation to outputs/metrics/{class_prefix}_metrics.json")

    print(f"OpenMIC samples: {len(samples)}")
    print(f"Known labels: {metrics['known_label_count']}")
    print(f"Positive samples: {metrics['num_positive_samples']}")
    print(f"Multi-positive samples: {metrics['num_multi_positive_samples']}")
    print(f"Device: {device}")
    print(f"micro_f1: {metrics['micro_f1']:.4f}")
    print(f"macro_f1: {metrics['macro_f1']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall: {metrics['recall']:.4f}")
    print(f"top1_positive_accuracy: {metrics['top1_positive_accuracy']:.4f}")
    print(f"masked_subset_accuracy: {metrics['masked_subset_accuracy']:.4f}")


if __name__ == "__main__":
    main()
