"""Fine-tune an IRMAS checkpoint on OpenMIC overlap-9 masked labels."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

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
from src.openmic import OpenMICDataset, OpenMICSample, discover_openmic_samples, openmic_distribution
from src.utils import (
    ensure_dir,
    ensure_output_dirs,
    get_checkpoint_model_size,
    get_device,
    get_state_dict_from_checkpoint,
    load_checkpoint,
    save_json,
    set_seed,
)


OVERLAP_INDICES = torch.tensor(config.OPENMIC_OVERLAP_CLASS_TO_IRMAS_INDEX, dtype=torch.long)


def split_samples(
    samples: list[OpenMICSample],
    validation_size: float,
    random_state: int,
) -> tuple[list[OpenMICSample], list[OpenMICSample]]:
    """Split OpenMIC train samples into fine-tune train and threshold-validation sets."""
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1")
    positive_counts = [min(int(sample.labels.sum()), 2) for sample in samples]
    train_samples, val_samples = train_test_split(
        samples,
        test_size=validation_size,
        random_state=random_state,
        shuffle=True,
        stratify=positive_counts,
    )
    return list(train_samples), list(val_samples)


def build_loader(
    samples: list[OpenMICSample],
    feature_type: str,
    batch_size: int,
    num_workers: int,
    cache_dir: Path,
    use_cache: bool,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    """Create a DataLoader for OpenMIC samples."""
    dataset = OpenMICDataset(
        samples,
        feature_type=feature_type,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def load_model(model_path: Path, device: torch.device) -> tuple[torch.nn.Module, str, dict]:
    """Load an IRMAS checkpoint for fine-tuning."""
    checkpoint = load_checkpoint(model_path, device)
    model_size = get_checkpoint_model_size(checkpoint)
    model = build_model(num_classes=config.NUM_CLASSES, model_size=model_size).to(device)
    model.load_state_dict(get_state_dict_from_checkpoint(checkpoint))
    return model, model_size, checkpoint if isinstance(checkpoint, dict) else {}


def compute_pos_weight(samples: list[OpenMICSample]) -> torch.Tensor:
    """Compute positive class weights over known overlap labels."""
    labels = np.stack([sample.labels for sample in samples], axis=0).astype(np.float32)
    masks = np.stack([sample.label_mask for sample in samples], axis=0).astype(bool)
    positive_counts = (labels * masks).sum(axis=0)
    known_counts = masks.sum(axis=0).astype(np.float32)
    negative_counts = known_counts - positive_counts
    return torch.from_numpy(negative_counts / np.maximum(positive_counts, 1.0)).float()


def masked_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
    pos_weight: torch.Tensor | None,
) -> torch.Tensor:
    """Compute BCE only on known OpenMIC overlap labels."""
    overlap_indices = OVERLAP_INDICES.to(logits.device)
    overlap_logits = logits.index_select(dim=1, index=overlap_indices)
    mask = label_mask.float()
    loss = F.binary_cross_entropy_with_logits(
        overlap_logits,
        labels.float(),
        pos_weight=None if pos_weight is None else pos_weight.to(logits.device),
        reduction="none",
    )
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    pos_weight: torch.Tensor | None,
    desc: str,
) -> float:
    """Run one train or validation epoch."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_known = 0.0

    for batch in tqdm(loader, desc=desc, leave=False):
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        label_mask = batch["label_mask"].to(device)

        if optimizer is not None:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(features)
            loss = masked_bce_loss(logits, labels, label_mask, pos_weight=pos_weight)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

        known_count = float(label_mask.float().sum().item())
        total_loss += float(loss.item()) * known_count
        total_known += known_count

    return total_loss / max(total_known, 1.0)


def save_checkpoint(
    model: torch.nn.Module,
    path: Path,
    model_size: str,
    args: argparse.Namespace,
    epoch: int,
    val_loss: float,
    history: list[dict],
    pos_weight: torch.Tensor | None,
    source_checkpoint: dict,
) -> None:
    """Save a fine-tuned 11-output checkpoint."""
    ensure_dir(path.parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_type": args.feature_type,
            "model_size": model_size,
            "class_codes": config.CLASS_CODES,
            "openmic_overlap_class_codes": config.OPENMIC_OVERLAP_CLASS_CODES,
            "openmic_overlap_class_names": [
                config.CLASS_NAMES[class_code] for class_code in config.OPENMIC_OVERLAP_CLASS_CODES
            ],
            "epoch": epoch,
            "val_loss": float(val_loss),
            "history": history,
            "pos_weight": None if pos_weight is None else pos_weight.detach().cpu().tolist(),
            "source_checkpoint": str(args.init_model_path),
            "source_checkpoint_epoch": source_checkpoint.get("epoch"),
            "train_args": {
                "openmic_dir": str(args.openmic_dir),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "feature_type": args.feature_type,
                "validation_size": args.validation_size,
                "random_state": args.random_state,
                "use_pos_weight": not args.no_pos_weight,
            },
        },
        path,
    )


def evaluate_with_thresholds(
    prefix: str,
    predictions: dict[str, np.ndarray | list[str]],
    samples: list[OpenMICSample],
    threshold_mode: str,
    selected_thresholds: float | dict[str, float],
    feature_type: str,
    split_name: str,
    num_examples: int,
) -> dict:
    """Evaluate and save predictions using fixed common or per-class thresholds."""
    probabilities = predictions["probabilities"]
    y_true = predictions["y_true"]
    label_mask = predictions["label_mask"]
    if threshold_mode == "common":
        y_pred = (probabilities >= float(selected_thresholds)).astype(np.int64)
        threshold_value = float(selected_thresholds)
    else:
        y_pred = apply_class_thresholds(probabilities, selected_thresholds)
        threshold_value = 0.0

    metrics, report_dict, report_text, tp_fp_fn_rows = compute_masked_metrics(
        y_true=y_true,
        y_pred=y_pred,
        label_mask=label_mask,
        probabilities=probabilities,
        threshold=threshold_value,
    )
    metrics.update(
        {
            "dataset": "openmic-2018",
            "split": split_name,
            "feature_type": feature_type,
            "num_eval_samples": len(samples),
            "threshold_mode": threshold_mode,
            "selected_thresholds": selected_thresholds,
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
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune IRMAS Mel v2 on OpenMIC overlap-9.")
    parser.add_argument("--openmic-dir", type=Path, default=config.DEFAULT_OPENMIC_DIR)
    parser.add_argument("--init-model-path", type=Path, default=default_model_path())
    parser.add_argument(
        "--output-model-path",
        type=Path,
        default=config.MODELS_DIR / "best_model_mel_v2_openmic_overlap9.pth",
    )
    parser.add_argument("--feature-type", choices=config.FEATURE_TYPES, default=config.DEFAULT_FEATURE_TYPE)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--random-state", type=int, default=config.RANDOM_STATE)
    parser.add_argument("--num-examples", type=int, default=30)
    parser.add_argument("--no-pos-weight", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=config.FEATURE_CACHE_DIR)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--prefix", default="openmic_overlap9_finetune")
    return parser.parse_args()


def main() -> None:
    """Fine-tune on OpenMIC train and evaluate fixed validation thresholds on test."""
    args = parse_args()
    ensure_output_dirs()
    set_seed(args.random_state)
    device = get_device()

    all_train_samples = discover_openmic_samples(
        data_dir=args.openmic_dir,
        split="train",
        positive_threshold=args.positive_threshold,
        require_known_overlap=True,
    )
    train_samples, val_samples = split_samples(
        all_train_samples,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )
    test_samples = discover_openmic_samples(
        data_dir=args.openmic_dir,
        split="test",
        positive_threshold=args.positive_threshold,
        require_known_overlap=True,
    )

    train_loader = build_loader(
        train_samples,
        args.feature_type,
        args.batch_size,
        args.num_workers,
        args.cache_dir,
        not args.no_cache,
        shuffle=True,
        device=device,
    )
    val_loader = build_loader(
        val_samples,
        args.feature_type,
        args.batch_size,
        args.num_workers,
        args.cache_dir,
        not args.no_cache,
        shuffle=False,
        device=device,
    )
    test_loader = build_loader(
        test_samples,
        args.feature_type,
        args.batch_size,
        args.num_workers,
        args.cache_dir,
        not args.no_cache,
        shuffle=False,
        device=device,
    )

    model, model_size, source_checkpoint = load_model(args.init_model_path, device)
    if (
        source_checkpoint
        and "feature_type" in source_checkpoint
        and source_checkpoint.get("feature_type") != args.feature_type
    ):
        warnings.warn(
            f"Checkpoint feature_type={source_checkpoint.get('feature_type')} "
            f"but fine-tuning uses {args.feature_type}",
            stacklevel=2,
        )

    pos_weight = None if args.no_pos_weight else compute_pos_weight(train_samples).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = 0

    print(f"Device: {device}")
    print(f"Model size: {model_size}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")
    if pos_weight is not None:
        print(f"Using OpenMIC overlap pos weights: {[round(v, 2) for v in pos_weight.detach().cpu().tolist()]}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer, pos_weight, desc="train")
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, device, None, pos_weight, desc="valid")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(
                model=model,
                path=args.output_model_path,
                model_size=model_size,
                args=args,
                epoch=epoch,
                val_loss=val_loss,
                history=history,
                pos_weight=pos_weight,
                source_checkpoint=source_checkpoint,
            )
            print(f"Saved best fine-tuned model to {args.output_model_path}")

    save_json(history, config.METRICS_DIR / f"{args.prefix}_training_history.json")
    checkpoint = load_checkpoint(args.output_model_path, device)
    model.load_state_dict(get_state_dict_from_checkpoint(checkpoint))

    val_predictions = collect_openmic_predictions(model, val_loader, device)
    val_sweep_rows = threshold_sweep(
        val_predictions["y_true"],
        val_predictions["probabilities"],
        val_predictions["label_mask"],
        default_thresholds(),
    )
    best_common = max(val_sweep_rows, key=lambda row: row["macro_f1"])
    best_common_threshold = float(best_common["threshold"])
    class_thresholds, class_threshold_rows = optimize_class_thresholds(
        val_predictions["y_true"],
        val_predictions["probabilities"],
        val_predictions["label_mask"],
        default_thresholds(),
    )
    save_dataframe_csv(
        pd.DataFrame(val_sweep_rows),
        config.METRICS_DIR / f"{args.prefix}_val_common_threshold_sweep.csv",
    )
    save_json(val_sweep_rows, config.METRICS_DIR / f"{args.prefix}_val_common_threshold_sweep.json")
    save_dataframe_csv(
        pd.DataFrame(class_threshold_rows),
        config.METRICS_DIR / f"{args.prefix}_val_class_thresholds.csv",
    )
    save_json(class_threshold_rows, config.METRICS_DIR / f"{args.prefix}_val_class_thresholds.json")
    save_json(
        {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_common_threshold": best_common_threshold,
            "best_common_val_metrics": best_common,
            "class_thresholds": class_thresholds,
        },
        config.METRICS_DIR / f"{args.prefix}_selected_thresholds.json",
    )

    val_common_metrics = evaluate_with_thresholds(
        prefix=f"{args.prefix}_val_common",
        predictions=val_predictions,
        samples=val_samples,
        threshold_mode="common",
        selected_thresholds=best_common_threshold,
        feature_type=args.feature_type,
        split_name="train_holdout_validation",
        num_examples=args.num_examples,
    )
    val_class_metrics = evaluate_with_thresholds(
        prefix=f"{args.prefix}_val_class_thresholds",
        predictions=val_predictions,
        samples=val_samples,
        threshold_mode="per_class",
        selected_thresholds=class_thresholds,
        feature_type=args.feature_type,
        split_name="train_holdout_validation",
        num_examples=args.num_examples,
    )

    test_predictions = collect_openmic_predictions(model, test_loader, device)
    test_common_metrics = evaluate_with_thresholds(
        prefix=f"{args.prefix}_test_common",
        predictions=test_predictions,
        samples=test_samples,
        threshold_mode="common",
        selected_thresholds=best_common_threshold,
        feature_type=args.feature_type,
        split_name="test",
        num_examples=args.num_examples,
    )
    test_class_metrics = evaluate_with_thresholds(
        prefix=f"{args.prefix}_test_class_thresholds",
        predictions=test_predictions,
        samples=test_samples,
        threshold_mode="per_class",
        selected_thresholds=class_thresholds,
        feature_type=args.feature_type,
        split_name="test",
        num_examples=args.num_examples,
    )

    save_json(
        {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "val_common": val_common_metrics,
            "val_class_thresholds": val_class_metrics,
            "test_common": test_common_metrics,
            "test_class_thresholds": test_class_metrics,
        },
        config.METRICS_DIR / f"{args.prefix}_summary.json",
    )

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(
        "Fine-tuned test common: "
        f"macro_f1={test_common_metrics['macro_f1']:.4f} | "
        f"micro_f1={test_common_metrics['micro_f1']:.4f}"
    )
    print(
        "Fine-tuned test class thresholds: "
        f"macro_f1={test_class_metrics['macro_f1']:.4f} | "
        f"micro_f1={test_class_metrics['micro_f1']:.4f} | "
        f"precision={test_class_metrics['precision']:.4f} | "
        f"recall={test_class_metrics['recall']:.4f}"
    )


if __name__ == "__main__":
    main()
