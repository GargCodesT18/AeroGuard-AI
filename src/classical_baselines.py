"""Train the two academic RUL baselines and compare them with the saved LSTM."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from data_loader import (
    ALL_DATASETS, create_sequences_per_engine, load_data, prepare_test_samples,
    prepare_train_data, risk_category, train_validation_split,
)
from evaluate import MODELS_DIR, RISK_LABELS, regression
from train import load_combined_train_data

REFERENCE = MODELS_DIR / "current"
OUTPUT = MODELS_DIR / "comparative"
STATISTICS = ("mean", "std", "min", "max", "latest", "slope")
CPU_THREADS = 4
FOREST = {
    "n_estimators": 96, "max_depth": 17, "min_samples_leaf": 6,
    "max_features": 0.5, "max_samples": 0.85, "n_jobs": CPU_THREADS,
}
MLP = {
    "hidden_layer_sizes": (128, 64), "activation": "relu", "solver": "adam",
    "alpha": 0.001, "learning_rate_init": 0.001, "batch_size": 512,
    "early_stopping": False, "shuffle": True,
}
MAX_EPOCHS = 100
PATIENCE = 12


def summarize_windows(windows: np.ndarray) -> np.ndarray:
    """Return six statistics per channel, in blocks ordered by STATISTICS."""
    n_windows, length, n_channels = windows.shape
    if length < 2:
        raise ValueError("A window needs two cycles to compute its slope")
    offsets = np.arange(length, dtype=np.float64) - (length - 1) / 2
    slope_weights = offsets / np.sum(offsets ** 2)
    result = np.empty((n_windows, len(STATISTICS) * n_channels), dtype=np.float32)
    for start in range(0, n_windows, 2048):
        batch = windows[start:start + 2048]
        result[start:start + len(batch)] = np.concatenate((
            batch.mean(axis=1, dtype=np.float64),
            batch.std(axis=1, dtype=np.float64),
            batch.min(axis=1),
            batch.max(axis=1),
            batch[:, -1, :],
            np.einsum("btc,t->bc", batch, slope_weights, dtype=np.float64),
        ), axis=1)
    return result


def training_data(config: dict):
    frame = load_combined_train_data(config["train_datasets"])
    frame, columns, target = prepare_train_data(frame, max_rul=config["max_rul"])
    if columns != config["feature_columns"]:
        raise ValueError("Training channels differ from the saved LSTM channel order")
    train_ids, validation_ids = train_validation_split(frame, config["validation_fraction"], config["seed"])
    if validation_ids != config["validation_ids"] or not set(train_ids).isdisjoint(validation_ids):
        raise ValueError("The reconstructed split differs from the saved LSTM split")

    def prepare(ids):
        windows, labels = create_sequences_per_engine(frame, columns, target, config["seq_length"], ids)
        return summarize_windows(windows), labels

    train_x, train_y = prepare(train_ids)
    val_x, val_y = prepare(validation_ids)
    split = {
        "training_engine_ids": train_ids, "validation_engine_ids": validation_ids,
        "training_windows": len(train_y), "validation_windows": len(val_y),
    }
    names = [f"{channel}_{stat}" for stat in STATISTICS for channel in columns]
    return train_x, train_y, val_x, val_y, names, split


def train_mlp(train_x, train_y, val_x, val_y, seed):
    started = perf_counter()
    scaler = StandardScaler()
    x = scaler.fit_transform(train_x)
    validation = scaler.transform(val_x)
    target_mean = float(train_y.mean(dtype=np.float64))
    target_std = float(train_y.std(dtype=np.float64))
    y = (train_y - target_mean) / target_std
    model = MLPRegressor(**MLP, random_state=np.random.RandomState(seed))
    best_model, best_epoch, best_rmse = None, 0, float("inf")
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.partial_fit(x, y)
        predicted = model.predict(validation) * target_std + target_mean
        rmse = float(np.sqrt(np.mean((predicted - val_y) ** 2)))
        if not np.isfinite(rmse):
            raise ValueError(f"MLP produced a non-finite validation RMSE at epoch {epoch}")
        history.append({"epoch": epoch, "train_loss_scaled": float(model.loss_), "validation_rmse": rmse})
        if rmse < best_rmse:
            best_model, best_epoch, best_rmse = deepcopy(model), epoch, rmse
        if epoch - best_epoch >= PATIENCE:
            break
        if epoch == 1 or epoch % 10 == 0:
            print(f"MLP epoch {epoch:3d}  validation RMSE {rmse:.3f}", flush=True)
    artifact = {
        "model": best_model, "scaler": scaler, "target_mean": target_mean, "target_std": target_std,
        "best_epoch": best_epoch, "best_validation_rmse": best_rmse,
    }
    return artifact, pd.DataFrame(history), perf_counter() - started


def evaluate_models(config, forest, mlp):
    frames = []
    for dataset in ALL_DATASETS:
        _, test, rul = load_data(dataset)
        windows, raw, ids = prepare_test_samples(
            test, rul, config["feature_columns"], seq_length=config["seq_length"], max_rul=None
        )
        x = summarize_windows(windows)
        frames.append(pd.DataFrame({
            "dataset": dataset, "engine_id": ids, "actual_raw": raw,
            "actual_capped": np.minimum(raw, config["max_rul"]),
            "Random Forest": forest.predict(x),
            "MLP": mlp["model"].predict(mlp["scaler"].transform(x)) * mlp["target_std"] + mlp["target_mean"],
        }))
    predictions = pd.concat(frames, ignore_index=True)
    reference = pd.read_csv(REFERENCE / "test_predictions.csv").rename(columns={"predicted": "LSTM"})
    merged = predictions.merge(
        reference[["dataset", "engine_id", "actual_raw", "actual_capped", "LSTM"]],
        on=["dataset", "engine_id"], how="outer", validate="one_to_one", indicator=True, suffixes=("", "_lstm"),
    )
    if len(merged) != 707 or (merged["_merge"] != "both").any():
        raise ValueError("The evaluated test engines differ from the saved LSTM cohort")
    for label in ("actual_raw", "actual_capped"):
        np.testing.assert_array_equal(merged[label], merged.pop(f"{label}_lstm"))
    return merged.drop(columns="_merge")


def score_models(predictions: pd.DataFrame, training: dict):
    scores, rows = {}, []
    groups = [*predictions.groupby("dataset", sort=True), ("pooled", predictions)]
    for name, details in training.items():
        scores[name] = {}
        for dataset, frame in groups:
            y_raw, y_capped, pred = (frame[col].to_numpy(dtype=np.float64) for col in ("actual_raw", "actual_capped", name))
            truth, risk = risk_category(y_raw), risk_category(pred)
            classes = classification_report(truth, risk, labels=RISK_LABELS, output_dict=True, zero_division=0)
            classes["accuracy"] = float(np.mean(truth == risk))
            classes["labels"] = RISK_LABELS
            classes["confusion_matrix"] = confusion_matrix(truth, risk, labels=RISK_LABELS).tolist()
            capped, uncapped = regression(y_capped, pred), regression(y_raw, pred)
            scores[name][dataset] = {
                "engines": len(frame), "capped": capped, "uncapped": uncapped,
                "risk": classes, "training_seconds": details["seconds"],
            }
            row = {"model": name, "dataset": dataset, "engines": len(frame), "training_seconds": details["seconds"]}
            for prefix, metrics in (("capped", capped), ("uncapped", uncapped)):
                row.update({f"{prefix}_{key}": metrics[key] for key in ("rmse", "mae", "r2", "nasa_score")})
            row["risk_accuracy"] = classes["accuracy"]
            row.update({f"risk_macro_{key}": value for key, value in classes["macro avg"].items() if key != "support"})
            rows.append(row)
    return scores, pd.DataFrame(rows).rename(columns={"risk_macro_f1-score": "risk_macro_f1"})


def save_plots(table, scores, importance, history, best_epoch):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    datasets = [*ALL_DATASETS, "pooled"]
    positions = np.arange(len(datasets))
    width = 0.24
    for ax, key, title, unit in zip(
        axes, ("capped_rmse", "uncapped_rmse", "risk_macro_f1"),
        ("Capped RUL", "Uncapped RUL", "Derived risk categories"),
        ("RMSE (cycles)", "RMSE (cycles)", "Macro F1"), strict=True,
    ):
        values = table.pivot(index="dataset", columns="model", values=key).reindex(datasets)
        for index, (model, color) in enumerate(zip(scores, ("#2864a6", "#b96319", "#23734d"))):
            ax.bar(positions + (index - 1) * width, values[model], width, label=model, color=color)
        ax.set_xticks(positions, datasets)
        ax.set(title=title, ylabel=unit)
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False)
    fig.savefig(OUTPUT / "model_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
    importance.head(16).iloc[::-1].plot.barh(x="feature", y="importance", ax=ax, legend=False)
    ax.set(title="Random Forest feature importance", xlabel="Mean decrease in impurity", ylabel="")
    fig.savefig(OUTPUT / "random_forest_feature_importance.png", dpi=150)
    plt.close(fig)

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    left.plot(history["epoch"], history["train_loss_scaled"])
    left.set(title="MLP training objective", xlabel="Epoch", ylabel="Loss (standardized RUL)")
    right.plot(history["epoch"], history["validation_rmse"])
    right.axvline(best_epoch, color="tab:orange", linestyle="--", label=f"Selected epoch {best_epoch}")
    right.set(title="Held-out engine validation", xlabel="Epoch", ylabel="RMSE (cycles)")
    right.legend()
    fig.savefig(OUTPUT / "mlp_learning_curves.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    for ax, (name, datasets) in zip(axes, scores.items()):
        matrix = np.asarray(datasets["pooled"]["risk"]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=707)
        for (row, col), count in np.ndenumerate(matrix):
            ax.text(col, row, str(count), ha="center", va="center")
        ax.set(title=name, xlabel="Predicted", ylabel="Actual")
        ax.set_xticks(range(3), RISK_LABELS, rotation=25)
        ax.set_yticks(range(3), RISK_LABELS)
    fig.savefig(OUTPUT / "pooled_confusion_matrices.png", dpi=150)
    plt.close(fig)


def main():
    report = json.loads((REFERENCE / "evaluation.json").read_text(encoding="utf-8"))
    for file, key in (("lstm_rul.pth", "checkpoint_sha256"), ("scaler.pkl", "scaler_sha256")):
        if hashlib.sha256((REFERENCE / file).read_bytes()).hexdigest() != report[key]:
            raise ValueError(f"The saved LSTM evaluation does not match {file}")
    config = report["config"]
    if config["seq_length"] != 50 or config["max_rul"] != 125 or tuple(config["train_datasets"]) != ALL_DATASETS:
        raise ValueError("The reference protocol must use FD001–FD004, 50-cycle windows and a 125-cycle cap")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    start = perf_counter()
    train_x, train_y, val_x, val_y, names, split = training_data(config)
    preparation_seconds = perf_counter() - start
    print(f"{len(train_y):,} train and {len(val_y):,} validation windows, {len(names)} inputs", flush=True)
    with threadpool_limits(limits=CPU_THREADS):
        print("Fitting Random Forest on training engines", flush=True)
        forest = RandomForestRegressor(**FOREST, random_state=config["seed"])
        start = perf_counter()
        forest.fit(train_x, train_y)
        forest_seconds = perf_counter() - start
        val_forest = float(np.sqrt(np.mean((forest.predict(val_x) - val_y) ** 2)))
        print(f"Random Forest finished in {forest_seconds:.1f}s. Validation RMSE {val_forest:.3f}", flush=True)
        print("Fitting MLP with held-out engine early stopping", flush=True)
        mlp, history, mlp_seconds = train_mlp(train_x, train_y, val_x, val_y, config["seed"])

        # Test data enters only after both estimators and the MLP epoch have been selected.
        predictions = evaluate_models(config, forest, mlp)

    training = {
        "LSTM": {"seconds": report["checkpoint_metrics"]["training_seconds"], "device": "CUDA",
                 "scope": "saved training loop, including validation", "best_epoch": report["checkpoint_metrics"]["best_epoch"],
                 "validation_rmse": report["checkpoint_metrics"]["best_val_rmse"]},
        "Random Forest": {"seconds": forest_seconds, "device": "CPU", "scope": "fit only",
                          "validation_rmse": val_forest, "configuration": FOREST},
        "MLP": {"seconds": mlp_seconds, "device": "CPU", "scope": "scaling, fit and validation selection",
                "validation_rmse": mlp["best_validation_rmse"], "best_epoch": mlp["best_epoch"],
                "epochs_run": len(history), "configuration": MLP, "max_epochs": MAX_EPOCHS, "patience": PATIENCE},
    }
    scores, table = score_models(predictions, training)
    for dataset, result in scores["LSTM"].items():
        stored = report["datasets"][dataset]
        for cap in ("capped", "uncapped"):
            np.testing.assert_allclose(
                [result[cap][key] for key in ("rmse", "mae", "r2", "nasa_score")],
                [stored[cap][key] for key in ("rmse", "mae", "r2", "nasa_score")], rtol=1e-10, atol=1e-10,
            )
        np.testing.assert_array_equal(result["risk"]["confusion_matrix"], stored["classification"]["confusion_matrix"])
    importance = pd.DataFrame({"feature": names, "importance": forest.feature_importances_})
    importance = importance.sort_values("importance", ascending=False, ignore_index=True)
    schema = {"feature_names": names, "channels": config["feature_columns"], "seq_length": config["seq_length"]}
    joblib.dump({"model": forest, **schema}, OUTPUT / "random_forest.joblib", compress=3)
    joblib.dump({**mlp, **schema}, OUTPUT / "mlp.joblib", compress=3)
    predictions.to_csv(OUTPUT / "test_predictions.csv", index=False)
    table.to_csv(OUTPUT / "comparison.csv", index=False)
    importance.to_csv(OUTPUT / "random_forest_feature_importance.csv", index=False)
    history.to_csv(OUTPUT / "mlp_history.csv", index=False)
    report = {
        "reference_checkpoint_sha256": report["checkpoint_sha256"], "sklearn_version": sklearn_version,
        "seed": config["seed"], "rul_cap": config["max_rul"], "cpu_threads": CPU_THREADS,
        "feature_preparation_seconds": preparation_seconds, "split": split, "features": schema,
        "training": training, "results": scores,
    }
    (OUTPUT / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    save_plots(table, scores, importance, history, mlp["best_epoch"])
    print(table[table["dataset"] == "pooled"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
