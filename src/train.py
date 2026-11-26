import argparse
import json
import os
import pickle
import random
from time import perf_counter
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import (
    ALL_DATASETS,
    DEFAULT_MAX_RUL,
    create_sequences_per_engine,
    load_data,
    load_training_data,
    nasa_score,
    prepare_test_data,
    prepare_train_data,
    scale_sequences,
    train_validation_split,
)
from model import LSTMRULPredictor

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM RUL model on NASA C-MAPSS data for cross-dataset generalization.")
    parser.add_argument("--train-datasets", nargs="+", default=list(ALL_DATASETS), help="Source dataset(s) for training (default: FD001 FD002 FD003 FD004)")
    parser.add_argument("--test-datasets", nargs="+", default=list(ALL_DATASETS), help="Target dataset(s) for testing (default: FD001 FD002 FD003 FD004)")
    parser.add_argument("--seq-length", type=int, default=50)
    parser.add_argument("--max-rul", type=int, default=DEFAULT_MAX_RUL)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR / "current")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--validation-only", action="store_true", help="Save the best checkpoint without running test inference")
    parser.add_argument("--schedule", action="store_true", help="Halve learning rate after two validation plateaus")
    parser.add_argument("--clip-grad", type=float, default=0, help="Maximum gradient norm (0 disables clipping)")
    parser.add_argument("--standardize-target", action="store_true", help="Train in standardized RUL units and export predictions in cycles")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "in-distribution", "cross-dataset", "multi-source"],
                        help="Experiment mode label. If auto, inferred from dataset args.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit_and_scale(X_train: np.ndarray, X_val: np.ndarray) -> StandardScaler:
    """Retain the upstream window-weighted fit, using only training windows."""
    scaler = StandardScaler()
    flat = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(flat)
    scaler.transform(flat, copy=False)
    scaler.transform(X_val.reshape(-1, X_val.shape[-1]), copy=False)
    return scaler


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_model(
    model: LSTMRULPredictor, loader: DataLoader, device: torch.device, target_mean: float = 0, target_std: float = 1
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    if len(loader) == 0:
        return 0.0, 0.0, 0.0, np.array([]), np.array([])

    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).cpu().numpy()
            preds.append(outputs)
            targets.append(y_batch.numpy())

    y_pred = np.concatenate(preds).reshape(-1) * target_std + target_mean
    y_true = np.concatenate(targets).reshape(-1) * target_std + target_mean
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    nasa = float(nasa_score(y_true, y_pred))
    return rmse, mae, nasa, y_true, y_pred


def load_combined_train_data(datasets: list[str]) -> pd.DataFrame:
    combined_train = []
    max_unit_id = 0
    for ds in datasets:
        train_df = load_training_data(ds)
        # Offset unit_id to prevent overlap between datasets
        train_df["unit_id"] += max_unit_id
        max_unit_id = train_df["unit_id"].max()
        combined_train.append(train_df)
    return pd.concat(combined_train, ignore_index=True)


def load_test_sets(datasets: list[str], feature_columns: list[str], seq_length: int, max_rul: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    test_sets = {}
    for ds in datasets:
        _, test_df, rul_df = load_data(ds)
        test_sets[ds] = prepare_test_data(test_df, rul_df, feature_columns, seq_length=seq_length, max_rul=max_rul)
    return test_sets


def plot_learning_curves(history: list[dict], out_path: Path):
    epochs = [item["epoch"] for item in history]
    train_rmse = [item["train_rmse"] for item in history]
    val_rmse = [item["val_rmse"] for item in history]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_rmse, label="Train RMSE", color="#58A6FF")
    plt.plot(epochs, val_rmse, label="Validation RMSE", color="#39D0D8")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (cycles)")
    plt.title("Training & Validation Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str, out_path: Path):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.7, color="#58A6FF")
    line_start = min(y_true.min(), y_pred.min())
    line_end = max(y_true.max(), y_pred.max())
    plt.plot([line_start, line_end], [line_start, line_end], "r--", linewidth=2)
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title(f"Predicted vs Actual RUL ({dataset_name})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_error_histogram(y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str, out_path: Path):
    errors = y_pred - y_true
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=30, color="#FF7B72", alpha=0.8, edgecolor="white")
    plt.axvline(0, color="w", linestyle="dashed", linewidth=2)
    plt.xlabel("Prediction Error (Predicted - Actual)")
    plt.ylabel("Frequency")
    plt.title(f"Prediction Error Histogram ({dataset_name})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    args.train_datasets = [d.upper() for d in args.train_datasets]
    args.test_datasets = [d.upper() for d in args.test_datasets]
    if (min(args.seq_length, args.max_rul, args.batch_size, args.epochs, args.patience) <= 0
            or not 0 < args.validation_fraction < 1 or args.clip_grad < 0):
        raise ValueError("Use positive training settings and a validation fraction between 0 and 1")
    if any(len(set(group)) != len(group) for group in (args.train_datasets, args.test_datasets)):
        raise ValueError("List each dataset only once per split")
    set_seed(args.seed)
    device = get_device()
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is unavailable. Install the CUDA PyTorch wheel and check the NVIDIA driver.")
    if device.type == "cuda":
        print(f"CUDA {torch.version.cuda}: {torch.cuda.get_device_name(device)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    unknown_train = [d for d in args.train_datasets if d not in ALL_DATASETS]
    unknown_test = [d for d in args.test_datasets if d not in ALL_DATASETS]
    if unknown_train or unknown_test:
        bad = sorted(set(unknown_train + unknown_test))
        raise ValueError(f"Unsupported dataset(s): {bad}. Allowed: {ALL_DATASETS}")

    mode = args.mode
    if mode == "auto":
        if set(args.train_datasets) == set(args.test_datasets) and len(args.train_datasets) == 1:
            mode = "in-distribution"
        elif len(args.train_datasets) > 1:
            mode = "multi-source"
        else:
            mode = "cross-dataset"

    print(f"--- Experiment Mode: {mode} ---")
    print(f"Source Training Datasets: {args.train_datasets}")
    print(f"Target Testing Datasets:  {args.test_datasets}")

    # Build combined train dataset
    train_df = load_combined_train_data(args.train_datasets)
    prepared_train_df, feature_columns, target = prepare_train_data(train_df, max_rul=args.max_rul)

    # Split train/val by unit_id to prevent leak
    train_ids, validation_ids = train_validation_split(
        prepared_train_df, validation_fraction=args.validation_fraction, random_state=args.seed
    )

    X_train, y_train = create_sequences_per_engine(
        prepared_train_df, feature_columns, target, seq_length=args.seq_length, unit_ids=train_ids
    )
    X_val, y_val = create_sequences_per_engine(
        prepared_train_df, feature_columns, target, seq_length=args.seq_length, unit_ids=validation_ids
    )

    if not len(X_train) or not len(X_val):
        raise ValueError("The sequence length leaves no training or validation windows")
    scaler = fit_and_scale(X_train, X_val)
    target_mean = float(y_train.mean()) if args.standardize_target else 0.0
    target_std = max(1.0, float(y_train.std())) if args.standardize_target else 1.0

    # Create loaders
    train_loader = make_loader(X_train, (y_train - target_mean) / target_std,
                               batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, (y_val - target_mean) / target_std,
                             batch_size=args.batch_size, shuffle=False)

    model = LSTMRULPredictor(
        input_size=len(feature_columns),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2) if args.schedule else None
    criterion = nn.MSELoss()

    history: list[dict] = []
    best_state = None
    best_val_rmse = float("inf")
    best_epoch = 0
    patience_counter = 0

    print(f"Training on {device} with {len(feature_columns)} features...")
    start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = torch.zeros((), device=device)
        learning_rate = optimizer.param_groups[0]["lr"]
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            if args.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            running_loss += loss.detach() * len(X_batch)

        train_rmse, train_mae, _, _, _ = evaluate_model(model, train_loader, device, target_mean, target_std)
        val_rmse, val_mae, _, _, _ = evaluate_model(model, val_loader, device, target_mean, target_std)
        epoch_loss = running_loss.item() / len(y_train) * target_std ** 2
        history.append({
            "epoch": epoch, "loss": epoch_loss,
            "train_rmse": train_rmse, "train_mae": train_mae,
            "val_rmse": val_rmse, "val_mae": val_mae, "learning_rate": learning_rate,
        })
        print(f"Epoch {epoch:02d}/{args.epochs} loss={epoch_loss:.4f} train_rmse={train_rmse:.2f} val_rmse={val_rmse:.2f}")
        if scheduler is not None:
            scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse, best_epoch, patience_counter = val_rmse, epoch, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training completed without producing a checkpoint.")

    elapsed = perf_counter() - start
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    print(f"Training loop {elapsed:.1f}s; peak CUDA tensor memory {peak_memory}")
    # Fold the inverse target transform into the existing linear head for exact serving parity.
    best_state["fc.weight"].mul_(target_std)
    best_state["fc.bias"].mul_(target_std).add_(target_mean)
    model.load_state_dict(best_state)

    # Plot learning curves
    plot_learning_curves(history, args.output_dir / "learning_curves.png")

    # Evaluate on all target test datasets
    test_metrics = {}
    print("\n" + "="*50)
    print(f"FINAL METRICS (Mode: {mode})")
    print("="*50)
    print(f"{'Dataset':<10} | {'RMSE':<8} | {'MAE':<8} | {'NASA':<10}")
    print("-" * 50)

    test_sets = {} if args.validation_only else load_test_sets(
        args.test_datasets, feature_columns, args.seq_length, args.max_rul
    )
    for ds_name, (X_test, y_test) in test_sets.items():
        X_test_scaled = scale_sequences(scaler, X_test)
        test_loader = make_loader(X_test_scaled, y_test, batch_size=args.batch_size, shuffle=False)
        rmse, mae, nasa, y_true, y_pred = evaluate_model(model, test_loader, device)
        test_metrics[ds_name] = {"rmse": rmse, "mae": mae, "nasa_score": nasa}
        print(f"{ds_name:<10} | {rmse:<8.2f} | {mae:<8.2f} | {nasa:<10.2f}")

        # Save plots for each target
        plot_predictions(y_true, y_pred, ds_name, args.output_dir / f"actual_vs_predicted_{ds_name}.png")
        plot_error_histogram(y_true, y_pred, ds_name, args.output_dir / f"error_histogram_{ds_name}.png")

    print("="*50)

    checkpoint = {
        "state_dict": best_state,
        "config": {
            "dataset": args.train_datasets[0] if len(args.train_datasets) == 1 else "MULTI",
            "train_datasets": args.train_datasets,
            "test_datasets": args.test_datasets,
            "mode": mode,
            "seq_length": args.seq_length,
            "max_rul": args.max_rul,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "input_size": len(feature_columns),
            "feature_columns": feature_columns,
            "seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "validation_ids": validation_ids,
            "target_mean": target_mean,
            "target_std": target_std,
            "training_options": {k: v for k, v in vars(args).items() if k != "output_dir"},
        },
        "metrics": {
            "best_epoch": best_epoch,
            "best_val_rmse": best_val_rmse,
            "test_metrics": test_metrics,
            "training_seconds": elapsed,
            "cuda_peak_bytes": peak_memory,
            "device": str(device),
            "torch_version": str(torch.__version__),
        },
        "history": history,
    }

    # Backward compatibility for api_server.py
    if test_metrics:
        first_test = args.test_datasets[0]
        checkpoint["metrics"].update({
            "test_rmse": test_metrics[first_test]["rmse"],
            "test_mae": test_metrics[first_test]["mae"],
            "test_nasa_score": test_metrics[first_test]["nasa_score"],
        })

    checkpoint_path = args.output_dir / "lstm_rul.pth"
    scaler_path = args.output_dir / "scaler.pkl"
    history_path = args.output_dir / "training_history.json"

    torch.save(checkpoint, checkpoint_path)
    with scaler_path.open("wb") as handle:
        pickle.dump(scaler, handle)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"\nSaved model:    {checkpoint_path}")
    print(f"Saved scaler:   {scaler_path}")


if __name__ == "__main__":
    main()
# COMMAND TO RUN TRAINING:
# python src/train.py --train-datasets FD001 FD002 FD003 FD004 --test-datasets FD001 FD002 FD003 FD004 --mode multi-source --epochs 20 --patience 5
