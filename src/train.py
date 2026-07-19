"""Training script for the IRMAS CNN baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.dataset import IRMASDataset, class_distribution, discover_samples, split_samples
from src.evaluate import build_prediction_rows, collect_predictions, compute_metrics, save_evaluation_files
from src.model import build_model
from src.utils import ensure_dir, ensure_output_dirs, get_device, load_checkpoint, save_json, set_seed


def compute_pos_weight(train_samples) -> torch.Tensor:
    """Compute positive class weights for BCEWithLogitsLoss."""
    positive_counts = torch.zeros(config.NUM_CLASSES, dtype=torch.float32)
    for sample in train_samples:
        class_index = config.CLASS_TO_INDEX[sample.class_code]
        positive_counts[class_index] += 1.0

    total_samples = float(len(train_samples))
    negative_counts = total_samples - positive_counts
    return negative_counts / positive_counts.clamp_min(1.0)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train a CNN instrument classifier on IRMAS.")
    parser.add_argument("--data-dir", type=Path, default=config.DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--feature-type", choices=config.FEATURE_TYPES, default=config.DEFAULT_FEATURE_TYPE)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--validation-size", type=float, default=config.VALIDATION_SIZE)
    parser.add_argument("--threshold", type=float, default=config.DEFAULT_THRESHOLD)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--random-state", type=int, default=config.RANDOM_STATE)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--model-size", choices=("small", "medium"), default="small")
    parser.add_argument("--num-examples", type=int, default=30)
    parser.add_argument("--no-pos-weight", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=config.FEATURE_CACHE_DIR)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--scheduler", action="store_true")
    return parser.parse_args()


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train the model for one epoch and return mean loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(dataloader, desc="train", leave=False):
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def validate_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate validation loss without updating weights."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="valid", leave=False):
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            logits = model(features)
            loss = criterion(logits, labels)

            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / max(total_samples, 1)


def save_training_plot(history: list[dict[str, float]], feature_type: str) -> Path:
    """Save train/validation loss plot."""
    output_path = config.FIGURES_DIR / f"loss_curve_{feature_type}.png"
    ensure_dir(output_path.parent)

    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="train loss")
    plt.plot(epochs, val_loss, marker="o", label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCEWithLogitsLoss")
    plt.title(f"Training history ({feature_type})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def save_model_checkpoint(
    model: torch.nn.Module,
    path: Path,
    args: argparse.Namespace,
    epoch: int,
    val_loss: float,
    history: list[dict[str, float]],
    pos_weight: torch.Tensor | None,
) -> None:
    """Save model weights and training metadata."""
    ensure_dir(path.parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_type": args.feature_type,
            "model_size": args.model_size,
            "class_codes": config.CLASS_CODES,
            "epoch": epoch,
            "val_loss": float(val_loss),
            "history": history,
            "pos_weight": None if pos_weight is None else pos_weight.detach().cpu().tolist(),
            "train_args": {
                "data_dir": str(args.data_dir),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "feature_type": args.feature_type,
                "model_size": args.model_size,
                "max_files": args.max_files,
                "validation_size": args.validation_size,
                "threshold": args.threshold,
                "random_state": args.random_state,
                "use_pos_weight": not args.no_pos_weight,
                "augment": args.augment,
                "scheduler": args.scheduler,
            },
        },
        path,
    )


def main() -> None:
    """Train the CNN model and save model, metrics and plots."""
    args = parse_args()
    ensure_output_dirs()
    if args.model_path is None:
        args.model_path = config.MODELS_DIR / f"best_model_{args.feature_type}.pth"
    set_seed(args.random_state)
    device = get_device()

    samples = discover_samples(args.data_dir, max_files=args.max_files)
    train_samples, val_samples = split_samples(
        samples,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )

    print(f"Device: {device}")
    print(f"Feature type: {args.feature_type}")
    print(f"Model size: {args.model_size}")
    print(f"Augmentation: {args.augment}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Total samples: {len(samples)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Train distribution: {class_distribution(train_samples)}")
    print(f"Validation distribution: {class_distribution(val_samples)}")

    train_dataset = IRMASDataset(
        train_samples,
        feature_type=args.feature_type,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        augment=args.augment,
    )
    val_dataset = IRMASDataset(
        val_samples,
        feature_type=args.feature_type,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(num_classes=config.NUM_CLASSES, model_size=args.model_size).to(device)
    pos_weight = None if args.no_pos_weight else compute_pos_weight(train_samples).to(device)
    if pos_weight is not None:
        print(f"Using BCE positive weights: {[round(value, 2) for value in pos_weight.detach().cpu().tolist()]}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        if args.scheduler
        else None
    )

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step(val_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
            }
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={learning_rate:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_model_checkpoint(model, args.model_path, args, epoch, val_loss, history, pos_weight)
            print(f"Saved best model to {args.model_path}")

    save_json(history, config.METRICS_DIR / f"training_history_{args.feature_type}.json")
    save_training_plot(history, args.feature_type)

    checkpoint = load_checkpoint(args.model_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions = collect_predictions(model, val_loader, device)
    probabilities = predictions["probabilities"]
    y_true = predictions["y_true"]
    y_pred = (probabilities >= args.threshold).astype("int64")
    metrics, report_dict, report_text, tp_fp_fn_rows = compute_metrics(
        y_true,
        y_pred,
        probabilities=probabilities,
        threshold=args.threshold,
    )
    metrics.update(
        {
            "feature_type": args.feature_type,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "num_train_samples": len(train_samples),
            "num_validation_samples": len(val_samples),
        }
    )
    prediction_rows = build_prediction_rows(
        predictions["paths"],
        y_true,
        y_pred,
        probabilities,
        max_rows=args.num_examples,
    )
    save_evaluation_files(
        prefix=f"validation_{args.feature_type}",
        metrics=metrics,
        report_dict=report_dict,
        report_text=report_text,
        tp_fp_fn_rows=tp_fp_fn_rows,
        prediction_rows=prediction_rows,
    )

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Validation micro_f1: {metrics['micro_f1']:.4f}")
    print(f"Validation macro_f1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
