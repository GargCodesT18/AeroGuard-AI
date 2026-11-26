from __future__ import annotations

import argparse
import json
import hashlib
from copy import deepcopy
from threading import Lock
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pickle
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Optional PyTorch import
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore

# Optional model imports
try:
    from data_loader import (
        DEFAULT_DATASET,
        load_data,
        prepare_test_frame,
        prepare_test_samples,
        nasa_score,
        risk_category,
        scale_sequences,
        prepare_train_data,
    )
    from model import LSTMRULPredictor, predict_with_uncertainty
    from evaluate import INTERVAL_LEVEL, default_models_dir, interval_bounds, trained_features
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False
    DEFAULT_DATASET = "FD001"
    LSTMRULPredictor = None  # type: ignore
    predict_with_uncertainty = None  # type: ignore
    load_data = None  # type: ignore
    prepare_test_samples = None  # type: ignore
    prepare_train_data = None  # type: ignore

# Optional import for feature importance
try:
    from captum.attr import IntegratedGradients  # type: ignore
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False
    IntegratedGradients = None  # type: ignore

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = default_models_dir() if HAS_MODEL else BASE_DIR / "models"
DASHBOARD_DIR = BASE_DIR / "aerospace-dashboard"


def get_device() -> Any:
    if not HAS_TORCH or torch is None:
        return None
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except (AttributeError, RuntimeError):
        pass
    return torch.device("cpu") if torch else None


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not HAS_TORCH or torch is None:
        return {
            "state_dict": None,
            "config": {
                "dataset": DEFAULT_DATASET,
                "seq_length": 50,
                "max_rul": 125,
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "feature_columns": None,
            },
            "metrics": {},
            "history": [],
        }

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return {
            "state_dict": None,
            "config": {
                "dataset": DEFAULT_DATASET,
                "seq_length": 50,
                "max_rul": 125,
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "feature_columns": None,
            },
            "metrics": {},
            "history": [],
        }

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint
    return {
        "state_dict": checkpoint,
        "config": {
            "dataset": DEFAULT_DATASET,
            "seq_length": 50,
            "max_rul": 125,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "feature_columns": None,
        },
        "metrics": {},
        "history": [],
    }


class ModelApiService:
    def __init__(self) -> None:
        self.checkpoint_path = MODELS_DIR / "lstm_rul.pth"
        self.scaler_path = MODELS_DIR / "scaler.pkl"
        self.history_path = MODELS_DIR / "training_history.json"
        self.device = get_device()
        self.available = True
        self.evaluation = {}
        self._mc_lock = Lock()

        # Check if required files exist and torch is available
        if not HAS_TORCH:
            print("WARNING: PyTorch not available - running in demo mode")
            self.available = False
            self.checkpoint = load_checkpoint(self.checkpoint_path)
            self.config = self.checkpoint["config"]
            self.metrics = {}
            self.history = []
            self.model = None
            self.scaler = None
            self.train_datasets = ["FD001"]
            self.dataset = "FD001"
            self.test_datasets = ["FD001"]
            self.mode = "demo"
            self.seq_length = 50
            self.max_rul = 125
            self._summary_cache = None
            self._test_cache = {}
            self._per_dataset_cache = {}
            return

        if not self.checkpoint_path.exists():
            print(f"WARNING: Missing checkpoint: {self.checkpoint_path}")
            self.available = False
            self.checkpoint = load_checkpoint(self.checkpoint_path)
            self.config = self.checkpoint["config"]
            self.metrics = {}
            self.history = []
            self.model = None
            self.scaler = None
            self.train_datasets = ["FD001"]
            self.dataset = "FD001"
            self.test_datasets = ["FD001"]
            self.mode = "demo"
            self.seq_length = 50
            self.max_rul = 125
            self._summary_cache = None
            self._test_cache = {}
            self._per_dataset_cache = {}
            return

        if not self.scaler_path.exists():
            print(f"WARNING: Missing scaler: {self.scaler_path}")
            self.available = False
            self.checkpoint = load_checkpoint(self.checkpoint_path)
            self.config = self.checkpoint["config"]
            self.metrics = {}
            self.history = []
            self.model = None
            self.scaler = None
            self.train_datasets = ["FD001"]
            self.dataset = "FD001"
            self.test_datasets = ["FD001"]
            self.mode = "demo"
            self.seq_length = 50
            self.max_rul = 125
            self._summary_cache = None
            self._test_cache = {}
            self._per_dataset_cache = {}
            return

        self.checkpoint = load_checkpoint(self.checkpoint_path)
        self.config = self.checkpoint["config"]
        self.metrics = self.checkpoint.get("metrics", {})
        train_datasets = self.config.get("train_datasets", [DEFAULT_DATASET])
        self.train_datasets = [d.upper() for d in train_datasets]
        if len(self.train_datasets) == 1:
            self.dataset = self.train_datasets[0]
        else:
            self.dataset = "MULTI"
        self.test_datasets = [d.upper() for d in self.config.get("test_datasets", self.train_datasets)]
        self.mode = self.config.get("mode", "in-distribution")
        self.seq_length = int(self.config.get("seq_length", 50))
        self.max_rul = int(self.config.get("max_rul", 125))

        self.history = self._load_history()
        self.model = self._load_model()
        self.mc_model = deepcopy(self.model)
        if self.mc_model is not None:
            self.mc_model.lstm.flatten_parameters()
        report_path = MODELS_DIR / "evaluation.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                report.get("checkpoint_sha256") == hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest()
                and report.get("scaler_sha256") == hashlib.sha256(self.scaler_path.read_bytes()).hexdigest()
            ):
                self.evaluation = report
        with self.scaler_path.open("rb") as handle:
            self.scaler = pickle.load(handle)

        self._summary_cache: dict[str, Any] | None = None
        self._test_cache = self._build_test_cache()
        self._per_dataset_cache: dict[str, dict] = {}
        if MODELS_DIR.name == "current":
            self._ensure_artifacts()

    def _load_history(self) -> list[dict[str, Any]]:
        if self.history_path.exists():
            return json.loads(self.history_path.read_text(encoding="utf-8"))
        return list(self.checkpoint.get("history", []))

    def _load_model(self) -> Any:
        if not HAS_MODEL or not HAS_TORCH or LSTMRULPredictor is None:
            return None

        if self.checkpoint["state_dict"] is None:
            self.available = False
            return None
        self.config.setdefault("input_size", self.checkpoint["state_dict"]["lstm.weight_ih_l0"].shape[1])
        feature_columns = trained_features(self.config)
        self.config["feature_columns"] = feature_columns
        model = LSTMRULPredictor(
            input_size=int(self.config.get("input_size", len(feature_columns))),
            hidden_size=int(self.config.get("hidden_size", 64)),
            num_layers=int(self.config.get("num_layers", 2)),
            dropout=float(self.config.get("dropout", 0.2)),
        ).to(self.device)
        model.load_state_dict(self.checkpoint["state_dict"])
        model.eval()
        return model

    def _build_test_cache(self, dataset: str | None = None) -> dict[str, Any]:
        if not self.available or self.scaler is None or self.model is None:
            return {}
        dataset = dataset or self.dataset
        try:
            train_df, test_df, rul_df = load_data(dataset)
            prepared_train_df, _, _ = prepare_train_data(train_df, max_rul=self.max_rul)
            features = self.config["feature_columns"]
            X_test, y_test, unit_ids = prepare_test_samples(
                test_df, rul_df, features, seq_length=self.seq_length, max_rul=self.max_rul
            )
            scaled = scale_sequences(self.scaler, X_test)
            tensor = torch.as_tensor(scaled, device=self.device)
            with torch.inference_mode():
                predictions = self.model(tensor).cpu().numpy().ravel()
        except (FileNotFoundError, ValueError) as exc:
            print(f"Data unavailable for {dataset}: {exc}")
            return {}
        return {
            "dataset": dataset,
            "train_df": train_df,
            "test_df": test_df,
            "rul_df": rul_df,
            "prepared_train_df": prepared_train_df,
            "prepared_test_df": prepare_test_frame(test_df, rul_df, self.max_rul),
            "feature_columns": features,
            "missing_features": [],
            "X_test": X_test,
            "X_test_scaled": scaled,
            "X_test_tensor": tensor,
            "y_test": y_test,
            "unit_ids": unit_ids,
            "predictions": predictions,
            "scaler": self.scaler,
            "use_original_scaler": True,
        }

    def _ensure_artifacts(self) -> None:
        if not self.available or not self._test_cache:
            return
        self._plot_training_history()
        self._plot_predictions_analysis()

    def _plot_training_history(self) -> None:
        if not self.history:
            return
        output_path = MODELS_DIR / "training_loss.png"
        epochs = [item["epoch"] for item in self.history]
        losses = [item["loss"] for item in self.history]
        val_rmse = [item["val_rmse"] for item in self.history]

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, losses, color="#58A6FF", linewidth=2, label="Training Loss (MSE)")
        plt.plot(epochs, val_rmse, color="#39D0D8", linewidth=2, label="Validation RMSE")
        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.title(f"Training Progress on NASA C-MAPSS ({self.dataset})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()

    def _plot_predictions_analysis(self) -> None:
        output_path = MODELS_DIR / "predictions_analysis.png"
        y_test = self._test_cache["y_test"]
        predictions = self._test_cache["predictions"]

        if predict_with_uncertainty is None or self._test_cache["X_test_tensor"] is None or self.model is None:
            return

        draw_count = self.evaluation.get("mc_samples", 20)
        with self._mc_lock:
            mean, std = predict_with_uncertainty(self.mc_model, self._test_cache["X_test_tensor"][:25], n_samples=draw_count)
        lower, upper, method = self._interval(mean.ravel(), std.ravel(), draw_count)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(y_test, predictions, alpha=0.7, s=40, color="#58A6FF")
        line_start = min(y_test.min(), predictions.min())
        line_end = max(y_test.max(), predictions.max())
        axes[0].plot([line_start, line_end], [line_start, line_end], "r--", linewidth=2)
        axes[0].set_xlabel("Actual RUL")
        axes[0].set_ylabel("Predicted RUL")
        axes[0].set_title("Predicted vs Actual RUL")
        axes[0].grid(True, alpha=0.3)

        sample_count = min(25, len(y_test))
        indices = np.arange(sample_count)
        axes[1].vlines(indices, lower, upper, color="#39D0D8", label=f"90% target · {method}")
        axes[1].scatter(indices, mean[:sample_count], color="#39D0D8", s=15, label="MC mean")
        axes[1].plot(indices, y_test[:sample_count], "r*", markersize=10, label="Actual RUL")
        axes[1].set_xlabel("Sample index")
        axes[1].set_ylabel("RUL")
        axes[1].set_title("Sample Predictions With Uncertainty")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close(fig)

    def get_feature_importance(self) -> dict[str, Any]:
        if not HAS_CAPTUM or IntegratedGradients is None or not HAS_TORCH or torch is None:
            return {"error": "Captum not installed"}

        X_test_tensor = self._test_cache.get("X_test_tensor")
        if X_test_tensor is None or self.model is None:
            return {"error": "Model or test data not available"}

        try:
            samples = X_test_tensor[:50].requires_grad_()
            ig = IntegratedGradients(self.model)  # type: ignore
            baseline = torch.zeros_like(samples)
            attributions, _ = ig.attribute(samples, baseline, return_convergence_delta=True)
            attr_mean = attributions.mean(dim=0).mean(dim=0).detach().cpu().numpy()

            feature_columns = self.config.get("feature_columns") or []
            if len(feature_columns) != len(attr_mean):
                return {"error": "Feature mismatch"}

            importance = {feat: float(abs(val)) for feat, val in zip(feature_columns, attr_mean)}
            return importance
        except Exception as e:
            return {"error": str(e)}

    def _compute_uncertainty_summary(self) -> dict[str, float]:
        if not self.available or predict_with_uncertainty is None or self._test_cache.get("X_test_tensor") is None or self.model is None:
            return {"averageStd": 0.0, "minStd": 0.0, "maxStd": 0.0}

        try:
            with self._mc_lock:
                mean_pred, std_pred = predict_with_uncertainty(
                    self.mc_model, self._test_cache["X_test_tensor"], n_samples=20
                )
            del mean_pred
            std_pred = std_pred.flatten()
            return {
                "averageStd": float(std_pred.mean()),
                "minStd": float(std_pred.min()),
                "maxStd": float(std_pred.max()),
            }
        except Exception:
            return {"averageStd": 0.0, "minStd": 0.0, "maxStd": 0.0}

    def get_summary(self) -> dict[str, Any]:
        if self._summary_cache is not None:
            return self._summary_cache

        train_df = self._test_cache["train_df"]
        test_df = self._test_cache["test_df"]
        feature_columns = self.config.get("feature_columns") or []
        uncertainty = self._compute_uncertainty_summary()

        evaluated = self.evaluation.get("datasets", {})
        test_metrics = {name: data["capped"] for name, data in evaluated.items() if name != "pooled"}
        test_metrics = test_metrics or self.metrics.get("test_metrics", {})
        if test_metrics:
            first_ds = list(test_metrics.keys())[0]
            primary_rmse = float(test_metrics[first_ds]["rmse"])
            primary_mae = float(test_metrics[first_ds]["mae"])
            primary_nasa = float(test_metrics[first_ds]["nasa_score"])
        else:
            primary_rmse = float(self.metrics.get("test_rmse", 0.0))
            primary_mae = float(self.metrics.get("test_mae", 0.0))
            primary_nasa = float(self.metrics.get("test_nasa_score", 0.0))

        summary = {
            "dataset": self.dataset,
            "device": str(self.device),
            "config": {
                "seqLength": self.seq_length,
                "maxRul": self.max_rul,
                "hiddenSize": int(self.config.get("hidden_size", 64)),
                "numLayers": int(self.config.get("num_layers", 2)),
                "dropout": float(self.config.get("dropout", 0.2)),
                "featureCount": len(feature_columns),
                "featureColumns": feature_columns,
            },
            "metrics": {
                "bestEpoch": int(self.metrics.get("best_epoch", 0)),
                "bestValRmse": float(self.metrics.get("best_val_rmse", 0.0)),
                "testRmse": primary_rmse,
                "testMae": primary_mae,
                "nasaScore": primary_nasa,
                "averageUncertainty": uncertainty["averageStd"],
                "uncertaintyMin": uncertainty["minStd"],
                "uncertaintyMax": uncertainty["maxStd"],
                "testRulCap": self.max_rul,
                "intervalEvaluation": evaluated.get("pooled", {}).get("uncertainty", {}).get("capped"),
                "intervalTarget": INTERVAL_LEVEL,
                "perDatasetMetrics": {k: {"rmse": float(v["rmse"]), "mae": float(v["mae"])} for k, v in test_metrics.items()},
            },
            "metadata": {
                "trainedOn": self.train_datasets,
                "validatedOn": self.train_datasets,
                "testedOn": self.test_datasets,
                "experimentMode": self.mode,
            },
            "data": {
                "trainRows": int(len(train_df)),
                "testRows": int(len(test_df)),
                "trainEngines": int(train_df["unit_id"].nunique()),
                "testEngines": int(test_df["unit_id"].nunique()),
                "operatingSettings": 3,
                "sensorCount": 21,
                "retainedFeatureCount": len(feature_columns),
            },
            "artifacts": {
                "trainingLossUrl": "/artifacts/training_loss.png",
                "predictionsAnalysisUrl": "/artifacts/predictions_analysis.png",
            },
            "sampleEngineIds": [int(x) for x in self._test_cache["unit_ids"][:10]],
        }
        self._summary_cache = summary
        return summary

    def get_history(self) -> list[dict[str, Any]]:
        return self.history

    def _interval(self, mean, std, mc_samples):
        calibration = self.evaluation.get("calibration", {})
        if calibration.get("mc_samples") == mc_samples:
            lower, upper = interval_bounds(mean, std, calibration["capped_multiplier"], calibration["min_std"])
            return lower, upper, "validation-scaled"
        lower, upper = interval_bounds(mean, std)
        return lower, upper, "uncalibrated dropout"

    def predict_sequence(
        self,
        sequence: np.ndarray,
        mc_samples: int = 20,
        scaler: Any | None = None,
    ) -> dict[str, Any]:
        effective_scaler = scaler or self.scaler
        if not self.available or not HAS_TORCH or torch is None or not HAS_MODEL or self.model is None or effective_scaler is None:
            return {"error": "Model or scaler not available"}

        feature_count = len(self.config["feature_columns"])
        if sequence.shape != (self.seq_length, feature_count):
            raise ValueError(
                f"Expected sequence shape {(self.seq_length, feature_count)}, got {tuple(sequence.shape)}"
            )

        try:
            if predict_with_uncertainty is None:
                return {"error": "predict_with_uncertainty not available"}

            scaled = effective_scaler.transform(sequence).reshape(1, self.seq_length, feature_count)
            tensor = torch.tensor(scaled, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                prediction = float(self.model(tensor).cpu().numpy().flatten()[0])
            with self._mc_lock:
                mean_pred, std_pred = predict_with_uncertainty(self.mc_model, tensor, n_samples=mc_samples)
            mean, std = float(mean_pred.item()), float(std_pred.item())
            lower, upper, method = self._interval(mean, std, mc_samples)
            return {
                "predictedRul": prediction,
                "meanPrediction": mean,
                "uncertaintyStd": std,
                "predictionInterval": {"lower": float(lower), "upper": float(upper),
                                       "target": INTERVAL_LEVEL, "method": method, "rulCap": self.max_rul},
                "mcSamples": mc_samples,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_sample_prediction(self, engine_id: int | None = None) -> dict[str, Any]:
        unit_ids = self._test_cache["unit_ids"]
        index = 0
        if engine_id is not None:
            matches = np.where(unit_ids == engine_id)[0]
            if len(matches) == 0:
                raise KeyError(f"Engine {engine_id} is not available in the test set.")
            index = int(matches[0])

        sequence = self._test_cache["X_test"][index]
        actual_rul = float(self._test_cache["y_test"][index])
        prediction = self.predict_sequence(sequence, mc_samples=20)
        prediction.update(
            {
                "engineId": int(unit_ids[index]),
                "actualRul": actual_rul,
                "absoluteError": abs(prediction["predictedRul"] - actual_rul),
            }
        )
        return prediction


app = Flask(__name__)
CORS(app)
service = ModelApiService()

def _get_dataset_cache(dataset: str) -> dict:
    """Use the checkpoint's source-training scaler for all target datasets."""
    dataset = dataset.upper()
    if dataset == service.dataset:
        cache = service._test_cache
    else:
        cache = service._per_dataset_cache.get(dataset)
        if cache is None:
            cache = service._build_test_cache(dataset)
            if cache:
                service._per_dataset_cache[dataset] = cache
    if not cache:
        raise FileNotFoundError(f"Model or data unavailable for {dataset}")
    return cache


@app.get("/api/summary")
def summary() -> Any:
    if not service._test_cache:
        return jsonify({"error": "Model or NASA C-MAPSS data unavailable"}), 503
    return jsonify(service.get_summary())


@app.get("/api/history")
def history() -> Any:
    dataset = request.args.get("dataset", service.dataset).upper()
    history = service.get_history()
    return jsonify(history)


@app.get("/api/sample-prediction")
def sample_prediction() -> Any:
    dataset  = request.args.get("dataset", service.dataset).upper()
    engine_id = request.args.get("engineId", type=int)
    use_random = request.args.get("random", "false").lower() == "true"
    try:
        cache = _get_dataset_cache(dataset)
        unit_ids = cache["unit_ids"]
        if use_random:
            index = int(np.random.randint(0, len(unit_ids)))
        elif engine_id is not None:
            matches = np.where(unit_ids == engine_id)[0]
            if len(matches) == 0:
                raise KeyError(f"Engine {engine_id} not found in {dataset}.")
            index = int(matches[0])
        else:
            index = 0

        sequence   = cache["X_test"][index]
        actual_rul = float(cache["y_test"][index])
        result = service.predict_sequence(sequence, mc_samples=service.evaluation.get("mc_samples", 20))
        result.update({
            "engineId":      int(unit_ids[index]),
            "actualRul":     actual_rul,
            "absoluteError": abs(result["predictedRul"] - actual_rul),
            "dataset":       dataset,
            "isFreshData":   dataset not in service.train_datasets,
        })
        return jsonify(result)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/all-predictions")
def all_predictions() -> Any:
    """Returns actual vs predicted RUL for scatter chart. Supports ?dataset=FD00X."""
    dataset = request.args.get("dataset", service.dataset).upper()
    try:
        cache = _get_dataset_cache(dataset)
        data = [
            {
                "engineId":    int(uid),
                "actualRul":   float(y),
                "predictedRul": float(p),
                "error":       float(abs(p - y)),
            }
            for uid, y, p in zip(cache["unit_ids"], cache["y_test"], cache["predictions"])
        ]
        rmse = float(np.sqrt(np.mean((cache["predictions"] - cache["y_test"]) ** 2)))
        mae  = float(np.mean(np.abs(cache["predictions"] - cache["y_test"])))
        return jsonify({"dataset": dataset, "rmse": round(rmse,4), "mae": round(mae,4), "engines": data})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/engine-ids")
def engine_ids() -> Any:
    """Returns sorted engine IDs for a given dataset. Supports ?dataset=FD00X."""
    dataset = request.args.get("dataset", service.dataset).upper()
    try:
        cache = _get_dataset_cache(dataset)
        uid_list = sorted(int(x) for x in set(cache["unit_ids"]))
        return jsonify(uid_list)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/explorer")
def explorer() -> Any:
    dataset = request.args.get("dataset", service.dataset).upper()
    limit = request.args.get("limit", default=25, type=int) or 25

    try:
        if not HAS_MODEL or load_data is None:
            return jsonify({"error": "Data loading not available"}), 503

        cache = _get_dataset_cache(dataset)
        test_df = cache["test_df"]
        latest_rows = test_df.sort_values(["unit_id", "cycle"]).groupby("unit_id").tail(1)

        prediction_by_engine = {
            int(uid): {"actualRul": float(actual), "predictedRul": float(predicted)}
            for uid, actual, predicted in zip(cache["unit_ids"], cache["y_test"], cache["predictions"])
        }

        explorer_rows: list[dict[str, Any]] = []
        for _, row in latest_rows.head(limit).iterrows():
            engine_id = int(row["unit_id"])
            prediction = prediction_by_engine[engine_id]
            predicted_rul = float(prediction["predictedRul"])
            actual_rul = float(prediction["actualRul"])
            health = max(0.0, min(100.0, (predicted_rul / service.max_rul) * 100.0))
            category = risk_category(predicted_rul).item()
            status = "NOMINAL" if category == "Healthy" else category.upper()

            engine_history = test_df[test_df["unit_id"] == engine_id].sort_values("cycle")
            trend_source = engine_history["sensor_2"].tail(8).to_numpy(dtype=np.float32)
            if len(trend_source) == 0:
                trend_source = np.zeros(8, dtype=np.float32)
            trend_min = float(trend_source.min())
            trend_max = float(trend_source.max())
            if trend_max - trend_min < 1e-8:
                trend = [50.0 for _ in trend_source]
            else:
                trend = [float(((value - trend_min) / (trend_max - trend_min)) * 100.0) for value in trend_source]

            explorer_rows.append(
                {
                    "engineId": engine_id,
                    "cycle": int(row["cycle"]),
                    "status": status,
                    "t2": float(row["sensor_2"]),
                    "p30": float(row["sensor_3"]),
                    "n1": float(row["sensor_4"]),
                    "vibe": float(row["sensor_11"]),
                    "health": round(health, 1),
                    "actualRul": round(actual_rul, 1),
                    "predictedRul": round(predicted_rul, 1),
                    "error": round(abs(predicted_rul - actual_rul), 1),
                    "trend": trend,
                }
            )

        summary = {
            "dataset": dataset,
            "engineCount": len(explorer_rows),
            "criticalCount": sum(1 for row in explorer_rows if row["status"] == "CRITICAL"),
            "warningCount": sum(1 for row in explorer_rows if row["status"] == "WARNING"),
            "averageHealth": round(float(np.mean([row["health"] for row in explorer_rows])) if explorer_rows else 0.0, 1),
        }

        return jsonify({"summary": summary, "rows": explorer_rows})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/notifications")
def notifications() -> Any:
    """Returns lightweight dashboard alerts derived from current model outputs."""
    dataset = request.args.get("dataset", service.dataset).upper()
    limit = request.args.get("limit", default=5, type=int) or 5

    try:
        cache = _get_dataset_cache(dataset)
        predictions = cache["predictions"]
        unit_ids = cache["unit_ids"]

        alerts: list[dict[str, Any]] = []
        for uid, pred in zip(unit_ids, predictions):
            predicted_rul = float(pred)
            level = risk_category(predicted_rul).item().lower()
            if level == "healthy":
                continue
            title = f"Engine {int(uid)} {level}"
            message = f"Simulated engine predicted RUL: {predicted_rul:.1f} cycles (illustrative threshold)."

            alerts.append(
                {
                    "id": f"{dataset}-{int(uid)}-{level}",
                    "level": level,
                    "title": title,
                    "message": message,
                    "dataset": dataset,
                    "createdAt": datetime.now(timezone.utc).isoformat(),
                }
            )

        alerts.sort(key=lambda item: 0 if item["level"] == "critical" else 1)
        alerts = alerts[: max(1, min(limit, 20))]

        if not alerts:
            return jsonify(
                {
                    "dataset": dataset,
                    "unreadCount": 0,
                    "notifications": [],
                    "checkedAt": datetime.now(timezone.utc).isoformat(),
                }
            )

        return jsonify(
            {
                "dataset": dataset,
                "unreadCount": len(alerts),
                "notifications": alerts,
                "checkedAt": datetime.now(timezone.utc).isoformat(),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/evaluate-fresh")
def evaluate_fresh() -> Any:
    """
    Runs the trained model against any C-MAPSS dataset.
    Uses the scaler fitted on the model's training engines.
    """
    dataset = request.args.get("dataset", "FD002").upper().strip()
    if dataset not in ("FD001", "FD002", "FD003", "FD004"):
        return jsonify({"error": "dataset must be one of FD001, FD002, FD003, FD004"}), 400

    try:
        cache = _get_dataset_cache(dataset)
        y_test = cache["y_test"]
        predictions = cache["predictions"]
        unit_ids = cache["unit_ids"]

        errors = np.abs(predictions - y_test)
        rmse = float(np.sqrt(np.mean((predictions - y_test) ** 2)))
        mae = float(np.mean(errors))
        nasa = nasa_score(y_test.astype(np.float64), predictions.astype(np.float64))

        per_engine = [
            {
                "engineId": int(uid),
                "actualRul": float(y),
                "predictedRul": float(p),
                "error": float(abs(p - y)),
            }
            for uid, y, p in zip(unit_ids, y_test, predictions)
        ]

        return jsonify({
            "dataset": dataset,
            "trainedOn": service.dataset,
            "engineCount": len(per_engine),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "nasaScore": round(nasa, 2),
            "rulCap": service.max_rul,
            "missingFeatures": cache.get("missing_features", []),
            "usedFeatures": cache.get("feature_columns", []),
            "predictions": per_engine,
        })

    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/predict")
def predict() -> Any:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or payload.get("sequence") is None:
        return jsonify({"error": "Missing sequence payload."}), 400
    if not service.available:
        return jsonify({"error": "Model unavailable"}), 503
    try:
        dataset = payload.get("dataset") or request.args.get("dataset") or service.dataset
        mc_samples = payload.get("mcSamples", service.evaluation.get("mc_samples", 20))
        if not isinstance(dataset, str) or dataset.upper() not in ("FD001", "FD002", "FD003", "FD004", "MULTI"):
            raise ValueError("Unsupported dataset")
        if type(mc_samples) is not int or not 2 <= mc_samples <= 1000:
            raise ValueError("mcSamples must be an integer from 2 through 1000")
        sequence_array = np.asarray(payload["sequence"], dtype=np.float32)
        if not np.isfinite(sequence_array).all():
            raise ValueError("Sequence values must be finite")
        result = service.predict_sequence(sequence_array, mc_samples=mc_samples)
        result["dataset"] = dataset.upper()
        return jsonify(result)
    except (ValueError, TypeError, OverflowError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/engine-history")
def engine_history() -> Any:
    """Returns the full historical telemetry and RUL predictions for a specific engine."""
    engine_id = request.args.get("engineId", type=int)
    dataset = request.args.get("dataset", service.dataset).upper()

    if engine_id is None:
        return jsonify({"error": "Missing engineId parameter"}), 400

    if not HAS_TORCH or torch is None or not service.available:
        return jsonify({"error": "Model not available"}), 503

    try:
        cache = _get_dataset_cache(dataset)
        prepared_test_df = cache.get("prepared_test_df")

        if prepared_test_df is None:
            return jsonify({"error": f"Prepared test data not available for {dataset}"}), 500

        engine_data = prepared_test_df[prepared_test_df["unit_id"] == engine_id].sort_values("cycle")
        if engine_data.empty:
            return jsonify({"error": f"Engine {engine_id} not found in {dataset}"}), 404

        cycles = engine_data["cycle"].tolist()
        actual_ruls = engine_data["RUL"].tolist()

        feature_cols = cache.get("feature_columns", [])
        raw_test_df = cache.get("test_df")
        raw_engine_data = None
        if raw_test_df is not None and engine_id in raw_test_df["unit_id"].values:
            raw_engine_data = raw_test_df[raw_test_df["unit_id"] == engine_id].sort_values("cycle")

        sensor_data = {}
        for col in feature_cols:
            if raw_engine_data is not None and col in raw_engine_data.columns:
                sensor_data[col] = raw_engine_data[col].tolist()
            elif col in engine_data.columns:
                sensor_data[col] = engine_data[col].tolist()
            else:
                sensor_data[col] = [0.0] * len(cycles)

        seq_length = service.seq_length
        n_cycles = len(cycles)
        step = 1 if n_cycles < 150 else 5
        step_indices = [i for i in range(n_cycles) if i % step == 0 or i == n_cycles - 1]
        n_pred = len(step_indices)

        raw_features = engine_data[feature_cols].to_numpy(dtype=np.float32)
        n_features = len(feature_cols)

        scaler = cache.get("scaler") or service.scaler

        if scaler is not None:
            raw_flat = raw_features.reshape(-1, n_features)
            scaled_flat = scaler.transform(raw_flat).reshape(n_cycles, n_features)
        else:
            scaled_flat = raw_features

        scaled_sequences = []
        for idx in step_indices:
            i = step_indices.index(idx)
            if idx + 1 < seq_length:
                sub_seq = scaled_flat[:idx+1]
                pad_rows = np.repeat(sub_seq[:1], seq_length - len(sub_seq), axis=0)
                sub_seq = np.vstack([pad_rows, sub_seq])
            else:
                sub_seq = scaled_flat[idx-seq_length+1:idx+1]
            scaled_sequences.append((i, idx, sub_seq))

        batch_tensor = torch.tensor(
            np.stack([s[2] for s in scaled_sequences]),
            dtype=torch.float32,
            device=service.device
        )
        with torch.no_grad():
            if service.model is None:
                return jsonify({"error": "Model is None"}), 503
            batch_preds = service.model(batch_tensor).cpu().numpy().flatten()

        predicted_ruls: list[float | None] = [None] * n_cycles
        for s, pred in zip(scaled_sequences, batch_preds):
            idx = int(s[1])
            if 0 <= idx < n_cycles:
                predicted_ruls[idx] = float(pred)

        last_idx = -1
        for i in range(n_cycles):
            if predicted_ruls[i] is not None:
                if last_idx != -1 and i - last_idx > 1:
                    start_val = predicted_ruls[last_idx]
                    end_val = predicted_ruls[i]
                    if start_val is not None and end_val is not None:
                        for j in range(last_idx + 1, i):
                            predicted_ruls[j] = start_val + (end_val - start_val) * (j - last_idx) / (i - last_idx)
                last_idx = i

        response = {
            "engineId": engine_id,
            "dataset": dataset,
            "cycles": cycles,
            "actualRul": actual_ruls,
            "predictedRul": predicted_ruls,
            "sensors": sensor_data
        }
        if request.args.get("includeInterval") == "1":
            draws = service.evaluation.get("mc_samples", 20)
            with service._mc_lock:
                mean, std = predict_with_uncertainty(service.mc_model, batch_tensor[-20:], n_samples=draws)
            lower, upper, method = service._interval(mean.ravel(), std.ravel(), draws)
            response["predictionInterval"] = {
                "cycles": [cycles[idx] for idx in step_indices[-20:]], "mean": mean.ravel().tolist(),
                "lower": lower.tolist(), "upper": upper.tolist(), "method": method, "target": INTERVAL_LEVEL,
            }
        return jsonify(response)

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/artifacts/<path:filename>")
def artifacts(filename: str) -> Any:
    if not (MODELS_DIR / filename).is_file() and filename == "training_loss.png":
        filename = "learning_curves.png"
    return send_from_directory(MODELS_DIR, filename)


@app.get("/")
def index() -> Any:
    return send_from_directory(DASHBOARD_DIR, "Main_Dashboard.html")


@app.get("/<path:filename>")
def static_pages(filename: str) -> Any:
    return send_from_directory(DASHBOARD_DIR, filename)


@app.get("/api/explain")
def api_explain() -> Any:
    def _format_sorted(importance_map: dict[str, float], method: str) -> Any:
        sorted_importance = sorted(importance_map.items(), key=lambda x: x[1], reverse=True)
        return jsonify({
            "features": [item[0] for item in sorted_importance],
            "values": [item[1] for item in sorted_importance],
            "method": method,
        })

    def _fallback_importance() -> Any:
        cache = service._test_cache
        # prefer explicit feature list from cache/config, but infer from prepared data if missing
        feature_columns = cache.get("feature_columns") or service.config.get("feature_columns") or []
        prepared_train_df = cache.get("prepared_train_df")
        prepared_test_df = cache.get("prepared_test_df")

        # If no explicit features provided, infer from prepared_train_df or prepared_test_df
        df_for_infer = prepared_train_df if prepared_train_df is not None else prepared_test_df
        if (not feature_columns or len(feature_columns) == 0) and df_for_infer is not None:
            # infer numeric feature columns excluding standard meta columns
            meta = {"RUL", "unit_id", "cycle", "max_cycle", "RUL_final"}
            feature_columns = [c for c in df_for_infer.columns if c not in meta]

        if df_for_infer is None or "RUL" not in df_for_infer.columns or len(feature_columns) == 0:
            return jsonify({"error": "Feature importance data unavailable"}), 503

        try:
            target = df_for_infer["RUL"].to_numpy(dtype=np.float64)
            if target.size == 0:
                return jsonify({"error": "No rows available for fallback importance"}), 503

            importance: dict[str, float] = {}
            target_std = float(np.std(target))
            for feature in feature_columns:
                if feature not in df_for_infer.columns:
                    continue
                col = df_for_infer[feature].to_numpy(dtype=np.float64)
                if col.size != target.size:
                    # If sizes mismatch, try to broadcast by truncation/expansion conservatively
                    min_len = min(col.size, target.size)
                    if min_len == 0:
                        continue
                    col = col[:min_len]
                    tgt = target[:min_len]
                else:
                    tgt = target

                col_std = float(np.std(col))
                if col_std <= 1e-12 or target_std <= 1e-12:
                    score = 0.0
                else:
                    corr = float(np.corrcoef(col, tgt)[0, 1])
                    score = abs(corr) if np.isfinite(corr) else 0.0
                importance[feature] = score

            if not importance:
                return jsonify({"error": "Unable to compute fallback feature importance"}), 503

            return _format_sorted(importance, "correlation_fallback")
        except Exception as exc:
            return jsonify({"error": f"fallback failed: {exc}"}), 500

    if HAS_CAPTUM and IntegratedGradients is not None and HAS_TORCH and torch is not None and service.available:
        X_test_tensor = service._test_cache.get("X_test_tensor")
        if X_test_tensor is None or service.model is None:
            return _fallback_importance()

        try:
            samples = X_test_tensor[:50].requires_grad_()
            ig = IntegratedGradients(service.model)  # type: ignore
            baseline = torch.zeros_like(samples)
            attributions, _ = ig.attribute(samples, baseline, return_convergence_delta=True)
            attr_mean = attributions.mean(dim=0).mean(dim=0).detach().cpu().numpy()

            feature_columns = service.config.get("feature_columns") or []
            if len(feature_columns) != len(attr_mean):
                return _fallback_importance()

            importance = {feat: float(abs(val)) for feat, val in zip(feature_columns, attr_mean)}
            return _format_sorted(importance, "integrated_gradients")
        except Exception:
            return _fallback_importance()

    return _fallback_importance()


@app.get("/api/latent-state")
def api_latent_state() -> Any:
    """Returns the hidden state activations of the LSTM for a sample sequence."""
    if not HAS_TORCH or torch is None or not service.available or service.model is None:
        return jsonify({"error": "Model not available"}), 503

    dataset = request.args.get("dataset", service.dataset).upper()
    try:
        cache = _get_dataset_cache(dataset)
        # Use first sample from test set
        X_tensor = cache.get("X_test_tensor")
        if X_tensor is None:
            X_tensor = cache["X_test_tensor"][:1] if "X_test_tensor" in cache else None
        else:
            X_tensor = X_tensor[:1]

        if X_tensor is None:
            return jsonify({"error": "Test tensor not available"}), 500

        # We need the LSTM output before the FC layer
        # Since we can't easily modify the forward without changing model.py,
        # we'll use a hook or just access the internal layers if possible.
        # Alternatively, just rerun the lstm part here.
        model = service.model
        with torch.no_grad():
            # Initial states
            h0 = torch.zeros(model.num_layers, X_tensor.size(0), model.hidden_size).to(X_tensor.device)
            c0 = torch.zeros(model.num_layers, X_tensor.size(0), model.hidden_size).to(X_tensor.device)

            # Forward pass through LSTM layer only
            out, _ = model.lstm(X_tensor, (h0, c0))
            # out shape: [batch, seq_len, hidden_size] -> [1, 50, 64]
            activations = out[0].cpu().numpy().tolist()

        return jsonify({
            "activations": activations, # 50 x 64
            "seq_length": service.seq_length,
            "hidden_size": model.hidden_size
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/accuracy-stats")
def api_accuracy_stats() -> Any:
    """Returns RMSE/MAE grouped by RUL buckets to show how accuracy changes near failure."""
    dataset = request.args.get("dataset", service.dataset).upper()
    try:
        cache = _get_dataset_cache(dataset)
        y_true = cache["y_test"]
        y_pred = cache["predictions"]

        buckets = [
            (0, 25, "0-25 (Critical)"),
            (25, 50, "25-50 (Late)"),
            (50, 80, "50-80 (Mid)"),
            (80, np.inf, "80+ (Early)")
        ]

        stats = []
        for low, high, label in buckets:
            mask = (y_true >= low) & (y_true < high)
            if np.any(mask):
                true_b = y_true[mask]
                pred_b = y_pred[mask]
                rmse = float(np.sqrt(np.mean((true_b - pred_b)**2)))
                mae = float(np.mean(np.abs(true_b - pred_b)))
                count = int(np.sum(mask))
            else:
                rmse, mae, count = 0.0, 0.0, 0
            stats.append({"label": label, "rmse": rmse, "mae": mae, "count": count})

        return jsonify(stats)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve the AeroGuard AI dashboard on localhost.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    app.run(
        host="127.0.0.1",
        port=args.port,
        debug=False,
        # Keep reloader off in this script entrypoint to avoid SystemExit
        # under debugpy/VS Code and single-process launcher confusion.
        use_reloader=False,
    )
