from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score

from data_loader import (
    ALL_DATASETS, CMAPSS_COLUMNS, DEFAULT_DATASET, load_data, nasa_score,
    prepare_test_samples, risk_category, scale_sequences,
)
from model import LSTMRULPredictor, predict_with_uncertainty
from train import get_device, load_combined_train_data, plot_error_histogram, plot_predictions, set_seed

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
INTERVAL_LEVEL = 0.90
NORMAL_MULTIPLIER = 1.6448536269514722
RISK_LABELS = ["Critical", "Warning", "Healthy"]


def default_models_dir() -> Path:
    current = MODELS_DIR / "current"
    return current if (current / "lstm_rul.pth").is_file() else MODELS_DIR


def parse_args() -> argparse.Namespace:
    defaults = default_models_dir()
    parser = argparse.ArgumentParser(description="Evaluate a saved AeroGuard AI checkpoint on NASA C-MAPSS.")
    parser.add_argument("--dataset", default="ALL", choices=[*ALL_DATASETS, "ALL"], type=str.upper)
    parser.add_argument("--checkpoint", type=Path, default=defaults / "lstm_rul.pth")
    parser.add_argument("--scaler", type=Path, help="Defaults to scaler.pkl beside the checkpoint")
    parser.add_argument("--output-dir", type=Path, help="Defaults to the checkpoint directory")
    parser.add_argument("--mc-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in checkpoint:
        return checkpoint
    return {
        "state_dict": checkpoint,
        "config": {
            "dataset": DEFAULT_DATASET, "train_datasets": [DEFAULT_DATASET], "test_datasets": [DEFAULT_DATASET],
            "seq_length": 50, "max_rul": 125, "hidden_size": 64, "num_layers": 2, "dropout": 0.2,
            "input_size": checkpoint["lstm.weight_ih_l0"].shape[1],
        },
        "metrics": {},
    }


def trained_features(config: dict) -> list[str]:
    features = config.get("feature_columns")
    if features:
        return list(features)
    if config.get("input_size", 24) != 24:
        raise ValueError("Legacy checkpoint is missing its input feature order")
    return CMAPSS_COLUMNS[2:]


def mc_predictions(model, sequences, device, mc_samples):
    means, stds = [], []
    for batch in np.array_split(sequences, max(1, (len(sequences) + 127) // 128)):
        tensor = torch.as_tensor(batch, device=device)
        mean, std = predict_with_uncertainty(model, tensor, n_samples=mc_samples)
        means.extend(mean.ravel())
        stds.extend(std.ravel())
    return np.asarray(means), np.asarray(stds)


def interval_bounds(mean, std, multiplier=NORMAL_MULTIPLIER, min_std=0.0):
    radius = multiplier * np.maximum(std, min_std)
    # RUL is non-negative. Apply the same bounds in evaluation and serving.
    return np.maximum(0, mean - radius), np.maximum(0, mean + radius)


def calibrate_intervals(model, config, scaler, device, mc_samples, seed):
    if not config.get("validation_ids"):
        return {}
    train = load_combined_train_data(config["train_datasets"])
    rng = np.random.default_rng(seed)
    windows, raw_labels = [], []
    seq_length = config["seq_length"]
    features = trained_features(config)
    # One pseudo-test endpoint per held-out engine. The original train/test sets remain untouched.
    held_out = train[train["unit_id"].isin(config["validation_ids"])]
    for _, engine in held_out.groupby("unit_id", sort=True):
        engine = engine.sort_values("cycle")
        stop = int(rng.integers(1, len(engine) + 1))
        sample = engine.iloc[max(0, stop - seq_length):stop][features].to_numpy(dtype=np.float32)
        windows.append(np.pad(sample, ((seq_length - len(sample), 0), (0, 0)), mode="edge"))
        raw_labels.append(len(engine) - stop)
    raw = np.asarray(raw_labels)
    mean, std = mc_predictions(model, scale_sequences(scaler, np.stack(windows)), device, mc_samples)
    cut = min(len(raw), int(np.ceil((len(raw) + 1) * INTERVAL_LEVEL))) - 1
    return {
        "engines": len(raw), "seed": seed, "mc_samples": mc_samples, "min_std": 1.0,
        "capped_multiplier": float(np.sort(abs(np.minimum(raw, config["max_rul"]) - mean) / np.maximum(std, 1))[cut]),
        "uncapped_multiplier": float(np.sort(abs(raw - mean) / np.maximum(std, 1))[cut]),
    }


def regression(y, pred):
    residual = pred - y
    return {
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "nasa_score": nasa_score(y.astype(np.float64), pred.astype(np.float64)),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "residual_quantiles_05_50_95": np.quantile(residual, [0.05, 0.5, 0.95]).tolist(),
        "overprediction_rate": float(np.mean(residual > 0)),
    }


def interval_metrics(y, lower, upper):
    return {
        "coverage": float(np.mean((lower <= y) & (y <= upper))),
        "average_width": float(np.mean(upper - lower)),
        "below_interval": float(np.mean(y < lower)),
        "above_interval": float(np.mean(y > upper)),
    }


def summarize(rows, calibration):
    raw, capped, predicted, means, stds = (
        np.array([row[key] for row in rows]) for key in ("actual_raw", "actual_capped", "predicted", "mc_mean", "mc_std")
    )
    actual_risk, predicted_risk = risk_category(raw), risk_category(predicted)
    cls = classification_report(actual_risk, predicted_risk, labels=RISK_LABELS, output_dict=True, zero_division=0)
    cls["accuracy"] = float(np.mean(actual_risk == predicted_risk))
    cls["labels"] = RISK_LABELS
    cls["confusion_matrix"] = confusion_matrix(actual_risk, predicted_risk, labels=RISK_LABELS).tolist()
    raw_bounds = interval_bounds(means, stds)
    uncertainty = {"mean_mc_std": float(stds.mean()), "nominal_level": INTERVAL_LEVEL}
    for name, y in (("capped", capped), ("uncapped", raw)):
        uncertainty[name] = {"dropout": interval_metrics(y, *raw_bounds)}
        if calibration:
            bounds = interval_bounds(means, stds, calibration[f"{name}_multiplier"], calibration["min_std"])
            uncertainty[name]["validation_scaled"] = interval_metrics(y, *bounds)
    return {
        "engines": len(rows), "labels_capped": int((raw != capped).sum()),
        "capped": regression(capped, predicted), "uncapped": regression(raw, predicted),
        "classification": cls, "uncertainty": uncertainty,
    }


def evaluate_dataset(dataset, model, config, scaler, device, mc_samples):
    _, test, rul = load_data(dataset)
    windows, labels, ids = prepare_test_samples(
        test, rul, trained_features(config), seq_length=config.get("seq_length", 50), max_rul=None
    )
    scaled = scale_sequences(scaler, windows)
    with torch.inference_mode():
        predicted = model(torch.as_tensor(scaled, device=device)).cpu().numpy().ravel()
    means, stds = mc_predictions(model, scaled, device, mc_samples)
    cap = config.get("max_rul", 125)
    return [
        {"dataset": dataset, "engine_id": int(uid), "actual_raw": float(y), "actual_capped": float(min(y, cap)),
         "predicted": float(p), "mc_mean": float(m), "mc_std": float(s)}
        for uid, y, p, m, s in zip(ids, labels, predicted, means, stds)
    ]


def main() -> None:
    args = parse_args()
    if args.mc_samples < 2:
        raise ValueError("--mc-samples requires at least two draws")
    set_seed(args.seed)
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    scaler_path = args.scaler or args.checkpoint.with_name("scaler.pkl")
    with scaler_path.open("rb") as handle:
        scaler = pickle.load(handle)
    device = get_device()
    model = LSTMRULPredictor(
        len(trained_features(config)), config.get("hidden_size", 64), config.get("num_layers", 2),
        config.get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    calibration = calibrate_intervals(model, config, scaler, device, args.mc_samples, args.seed + 1)
    # Reset RNG so the derived uncertainty for the test cohort does not depend on calibration.
    set_seed(args.seed)
    datasets = ALL_DATASETS if args.dataset == "ALL" else (args.dataset,)
    output = args.output_dir or args.checkpoint.parent
    output.mkdir(parents=True, exist_ok=True)
    rows, results = [], {}
    for dataset in datasets:
        predictions = evaluate_dataset(dataset, model, config, scaler, device, args.mc_samples)
        rows.extend(predictions)
        results[dataset] = summarize(predictions, calibration)
        y = np.array([row["actual_capped"] for row in predictions])
        p = np.array([row["predicted"] for row in predictions])
        plot_predictions(y, p, f"{dataset}, capped", output / f"actual_vs_predicted_{dataset}.png")
        plot_error_histogram(y, p, f"{dataset}, capped", output / f"error_histogram_{dataset}.png")
    results["pooled"] = summarize(rows, calibration)
    report = {
        "checkpoint": args.checkpoint.name, "device": str(device), "seed": args.seed, "mc_samples": args.mc_samples,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "scaler_sha256": hashlib.sha256(scaler_path.read_bytes()).hexdigest(),
        "config": config, "checkpoint_metrics": checkpoint.get("metrics", {}),
        "calibration": calibration, "datasets": results,
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Evaluation on {device} | n | capped RMSE | raw RMSE | capped MAE | capped R2 | macro F1")
    for dataset, result in results.items():
        capped = result["capped"]
        print(f"{dataset:<6} {result['engines']:4} {capped['rmse']:12.3f} {result['uncapped']['rmse']:10.3f} "
              f"{capped['mae']:12.3f} {capped['r2']:11.3f} {result['classification']['macro avg']['f1-score']:10.3f}")
    print(f"Full evaluation and engine-level predictions: {output}")


if __name__ == "__main__":
    main()
