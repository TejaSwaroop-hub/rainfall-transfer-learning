#!/usr/bin/env python3
"""Train spatially informed ERA5->IMD GRUs.

Adds normalized latitude and longitude to each of the 7 input timesteps
(15 atmospheric features + 2 static location features = 17 inputs).

Experiments:
  1) ERA5 spatial GRU pretraining (2014-2017), validation (2018)
  2) Spatial scratch GRU on IMD (2019-2021), validation (2022)
  3) Spatial transfer GRU initialized from the spatial ERA5 pretrained encoder

2023 was previously inspected by the researcher. Its evaluation is therefore
labelled exploratory/post-hoc, not a fresh untouched test.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
NPZ_PATH = ROOT / "data" / "processed" / "ml_sequences_2014_2023.npz"
BASE_CHECKPOINT = ROOT / "results" / "checkpoints" / "era5_pretrained_gru.pt"
CHECKPOINT_DIR = ROOT / "results" / "checkpoints"
METRICS_DIR = ROOT / "results" / "metrics"
PREDICTIONS_DIR = ROOT / "results" / "predictions"
FIGURES_DIR = ROOT / "results" / "figures"
LOG_PATH = ROOT / "logs" / "spatial_gru_training.log"

SEED = 142
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 40
PATIENCE = 7
EVENT_THRESHOLD_MM = 2.5
LOOKBACK = 7
ATMOSPHERIC_FEATURES = 15
INPUT_SIZE = 17


def configure_logging() -> None:
    for directory in (CHECKPOINT_DIR, METRICS_DIR, PREDICTIONS_DIR, FIGURES_DIR, LOG_PATH.parent):
        directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH, mode="w")],
        force=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_dict_from_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Required original checkpoint is missing: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("model_state_dict"), dict):
        return payload["model_state_dict"]
    if isinstance(payload, dict) and all(isinstance(v, torch.Tensor) for v in payload.values()):
        return payload
    raise ValueError("Could not identify model_state_dict in original ERA5 checkpoint.")


def infer_hidden_size(state: dict[str, torch.Tensor]) -> int:
    key = "gru.weight_hh_l0"
    if key not in state:
        raise KeyError(f"Expected {key} in original checkpoint; keys include {list(state)[:8]}")
    return int(state[key].shape[1])


class SpatialGRURegressor(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.gru(x)
        return self.head(sequence_output[:, -1, :]).squeeze(-1)


def add_spatial_features(
    X: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    coord_mean: np.ndarray,
    coord_std: np.ndarray,
    split_name: str,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    lat = np.asarray(latitude, dtype=np.float32).reshape(-1)
    lon = np.asarray(longitude, dtype=np.float32).reshape(-1)
    if X.ndim != 3 or X.shape[1:] != (LOOKBACK, ATMOSPHERIC_FEATURES):
        raise ValueError(f"{split_name}: expected X shape (N, 7, 15); got {X.shape}")
    if len(X) != len(lat) or len(X) != len(lon):
        raise ValueError(f"{split_name}: feature and coordinate sample counts differ")
    coords = np.column_stack([lat, lon]).astype(np.float32)
    coords = (coords - coord_mean) / coord_std
    coords_seq = np.repeat(coords[:, None, :], LOOKBACK, axis=1)
    X17 = np.concatenate([X, coords_seq], axis=2).astype(np.float32)
    if not np.isfinite(X17).all():
        raise ValueError(f"{split_name}: non-finite values after adding coordinates")
    return X17


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool, seed: int, device: torch.device) -> DataLoader:
    xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32).reshape(-1))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(xt, yt),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    label: str,
    checkpoint_path: Path,
    seed: int,
) -> tuple[nn.Module, list[dict[str, float]], int, float]:
    set_seed(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_sum += float(loss.item()) * len(xb)
            train_count += len(xb)

        model.eval()
        val_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                loss = criterion(model(xb), yb)
                val_sum += float(loss.item()) * len(xb)
                val_count += len(xb)

        train_loss = train_sum / max(train_count, 1)
        val_loss = val_sum / max(val_count, 1)
        history.append({"model": label, "epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
        logging.info("%s | epoch %02d/%02d | train MSE(log1p)=%.6f | val MSE(log1p)=%.6f",
                     label, epoch, MAX_EPOCHS, train_loss, val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                logging.info("%s | early stopping after epoch %d", label, epoch)
                break

    if best_state is None:
        raise RuntimeError(f"No best checkpoint recorded for {label}")
    model.load_state_dict(best_state)
    checkpoint = {
        "model_state_dict": best_state,
        "model_label": label,
        "input_size": INPUT_SIZE,
        "hidden_size": int(model.gru.hidden_size),
        "lookback": LOOKBACK,
        "feature_count": INPUT_SIZE,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_mse_log1p": best_val,
        "coordinate_mean": coord_mean_global.tolist(),
        "coordinate_std": coord_std_global.tolist(),
        "description": "15 standardized ERA5 atmospheric predictors plus standardized latitude and longitude repeated across the 7 input timesteps.",
    }
    torch.save(checkpoint, checkpoint_path)
    logging.info("%s | best epoch=%d | best validation MSE(log1p)=%.6f | saved=%s",
                 label, best_epoch, best_val, checkpoint_path)
    return model, history, best_epoch, best_val


def predict_log(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    loader = DataLoader(torch.from_numpy(np.asarray(X, dtype=np.float32)), batch_size=1024, shuffle=False)
    with torch.no_grad():
        for xb in loader:
            outputs.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def invert_log_prediction(pred_log: np.ndarray) -> np.ndarray:
    return np.maximum(np.expm1(np.maximum(pred_log, 0.0)), 0.0)


def kge_score(obs: np.ndarray, pred: np.ndarray) -> float:
    obs_mean = float(np.mean(obs))
    pred_mean = float(np.mean(pred))
    obs_std = float(np.std(obs))
    pred_std = float(np.std(pred))
    if obs_std == 0 or obs_mean == 0:
        return float("nan")
    corr = float(np.corrcoef(obs, pred)[0, 1]) if obs_std > 0 and pred_std > 0 else 0.0
    alpha = pred_std / obs_std
    beta = pred_mean / obs_mean
    return float(1.0 - np.sqrt((corr - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def metric_row(name: str, obs: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    error = pred - obs
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    denominator = float(np.sum((obs - np.mean(obs)) ** 2))
    nse = float(1.0 - np.sum(error ** 2) / denominator) if denominator > 0 else float("nan")
    actual_event = obs >= EVENT_THRESHOLD_MM
    predicted_event = pred >= EVENT_THRESHOLD_MM
    tp = int(np.sum(actual_event & predicted_event))
    fp = int(np.sum(~actual_event & predicted_event))
    fn = int(np.sum(actual_event & ~predicted_event))
    tn = int(np.sum(~actual_event & ~predicted_event))
    pod = tp / (tp + fn) if tp + fn else float("nan")
    far = fp / (tp + fp) if tp + fp else float("nan")
    csi = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = (2 * (tp * tn - fp * fn) / hss_den) if hss_den else float("nan")
    return {
        "model": name, "n_samples": len(obs), "MAE_mm": mae, "RMSE_mm": rmse,
        "NSE": nse, "KGE": kge_score(obs, pred), "event_threshold_mm": EVENT_THRESHOLD_MM,
        "POD": pod, "FAR": far, "CSI": csi, "HSS": hss,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def plot_history(histories: dict[str, list[dict[str, float]]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, key, title in zip(
        axes,
        ("spatial_source", "spatial_scratch", "spatial_transfer"),
        ("ERA5 spatial pretraining", "Spatial scratch GRU on IMD", "Spatial transfer GRU on IMD"),
    ):
        hist = histories[key]
        epochs = [int(r["epoch"]) for r in hist]
        ax.plot(epochs, [r["train_loss"] for r in hist], marker="o", label="Training")
        ax.plot(epochs, [r["val_loss"] for r in hist], marker="s", label="Validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE on log1p rainfall")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Spatially informed GRU: training and validation curves")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "spatial_gru_training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(obs: np.ndarray, predictions: dict[str, np.ndarray]) -> None:
    max_obs = max(float(np.max(obs)), 1.0)
    max_axis = max_obs * 1.02
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for key, label in (("spatial_scratch", "Spatial scratch GRU"), ("spatial_transfer", "Spatial transfer GRU")):
        axes[0].scatter(obs, predictions[key], s=8, alpha=0.20, label=label)
    axes[0].plot([0, max_axis], [0, max_axis], "k--", label="Ideal")
    axes[0].set(xlim=(0, max_axis), ylim=(0, max_axis), xlabel="Observed IMD rainfall (mm/day)", ylabel="Predicted rainfall (mm/day)", title="2023 exploratory evaluation: full range")
    axes[0].legend()
    zoom = obs <= np.quantile(obs, 0.99)
    zoom_max = max(float(np.max(obs[zoom])), 1.0)
    for key, label in (("spatial_scratch", "Spatial scratch GRU"), ("spatial_transfer", "Spatial transfer GRU")):
        axes[1].scatter(obs[zoom], predictions[key][zoom], s=8, alpha=0.20, label=label)
    axes[1].plot([0, zoom_max], [0, zoom_max], "k--", label="Ideal")
    axes[1].set(xlim=(0, zoom_max), ylim=(0, zoom_max), xlabel="Observed IMD rainfall (mm/day)", ylabel="Predicted rainfall (mm/day)", title="2023 exploratory evaluation: up to 99th percentile")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "spatial_gru_test_scatter_2023.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    global coord_mean_global, coord_std_global
    configure_logging()
    if not NPZ_PATH.exists():
        raise FileNotFoundError(NPZ_PATH)
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Loading prepared sequence data: %s", NPZ_PATH)
    logging.info("Device: %s", device)
    if device.type == "cuda":
        logging.info("GPU: %s", torch.cuda.get_device_name(0))

    with np.load(NPZ_PATH) as z:
        required = [
            "X_source_train", "y_source_train_log1p", "latitude_source_train", "longitude_source_train",
            "X_source_val", "y_source_val_log1p", "latitude_source_val", "longitude_source_val",
            "X_target_train", "y_target_train_log1p", "y_target_train_mm", "latitude_target_train", "longitude_target_train",
            "X_target_val", "y_target_val_log1p", "y_target_val_mm", "latitude_target_val", "longitude_target_val",
            "X_target_test", "y_target_test_mm", "dates_target_test", "latitude_target_test", "longitude_target_test",
        ]
        missing = [k for k in required if k not in z.files]
        if missing:
            raise KeyError(f"Missing NPZ arrays: {missing}")
        a = {k: np.asarray(z[k]) for k in required}

    # Fit coordinate scaling on source-training samples only. Same fixed 36-cell grid
    # is used for source and target periods; no target labels enter this calculation.
    coord_source = np.column_stack([a["latitude_source_train"], a["longitude_source_train"]]).astype(np.float64)
    coord_mean_global = coord_source.mean(axis=0)
    coord_std_global = coord_source.std(axis=0)
    coord_std_global[coord_std_global == 0] = 1.0
    logging.info("Coordinate scaler fitted on source training period only: mean=%s std=%s",
                 np.round(coord_mean_global, 5).tolist(), np.round(coord_std_global, 5).tolist())

    Xs_tr = add_spatial_features(a["X_source_train"], a["latitude_source_train"], a["longitude_source_train"], coord_mean_global, coord_std_global, "source_train")
    Xs_va = add_spatial_features(a["X_source_val"], a["latitude_source_val"], a["longitude_source_val"], coord_mean_global, coord_std_global, "source_val")
    Xt_tr = add_spatial_features(a["X_target_train"], a["latitude_target_train"], a["longitude_target_train"], coord_mean_global, coord_std_global, "target_train")
    Xt_va = add_spatial_features(a["X_target_val"], a["latitude_target_val"], a["longitude_target_val"], coord_mean_global, coord_std_global, "target_val")
    Xt_te = add_spatial_features(a["X_target_test"], a["latitude_target_test"], a["longitude_target_test"], coord_mean_global, coord_std_global, "target_test")
    logging.info("Spatial input shapes: source_train=%s target_train=%s (7 days x 17 features)", Xs_tr.shape, Xt_tr.shape)

    # Infer same hidden size as the original model so the only intended input change
    # is the addition of coordinate channels.
    old_state = state_dict_from_checkpoint(BASE_CHECKPOINT)
    hidden_size = infer_hidden_size(old_state)
    logging.info("Using original GRU hidden size: %d", hidden_size)

    histories: dict[str, list[dict[str, float]]] = {}
    y_source_tr = np.asarray(a["y_source_train_log1p"], dtype=np.float32)
    y_source_va = np.asarray(a["y_source_val_log1p"], dtype=np.float32)
    y_target_tr = np.asarray(a["y_target_train_log1p"], dtype=np.float32)
    y_target_va = np.asarray(a["y_target_val_log1p"], dtype=np.float32)

    # 1) Pretrain a new 17-input spatial GRU on ERA5, then validate on 2018.
    set_seed(SEED)
    source_model = SpatialGRURegressor(hidden_size)
    source_model, histories["spatial_source"], source_epoch, source_val_loss = fit_model(
        source_model,
        make_loader(Xs_tr, y_source_tr, True, SEED, device),
        make_loader(Xs_va, y_source_va, False, SEED, device),
        device, "ERA5 spatial-pretrained GRU", CHECKPOINT_DIR / "era5_spatial_pretrained_gru.pt", SEED,
    )

    # 2) Train a target model from scratch using IMD labels.
    set_seed(SEED + 1)
    scratch_model = SpatialGRURegressor(hidden_size)
    scratch_model, histories["spatial_scratch"], scratch_epoch, scratch_val_loss = fit_model(
        scratch_model,
        make_loader(Xt_tr, y_target_tr, True, SEED + 1, device),
        make_loader(Xt_va, y_target_va, False, SEED + 1, device),
        device, "Spatial scratch GRU on IMD", CHECKPOINT_DIR / "spatial_scratch_gru_imd.pt", SEED + 1,
    )

    # 3) Initialize target spatial GRU encoder with spatial ERA5 pretrained weights.
    set_seed(SEED + 1)
    transfer_model = SpatialGRURegressor(hidden_size)
    transfer_model.gru.load_state_dict(copy.deepcopy(source_model.gru.state_dict()), strict=True)
    transfer_model, histories["spatial_transfer"], transfer_epoch, transfer_val_loss = fit_model(
        transfer_model,
        make_loader(Xt_tr, y_target_tr, True, SEED + 1, device),
        make_loader(Xt_va, y_target_va, False, SEED + 1, device),
        device, "Spatial transfer GRU on IMD", CHECKPOINT_DIR / "spatial_transfer_gru_imd.pt", SEED + 1,
    )

    # Direct zero-shot source model on ERA5 features, evaluated against IMD labels.
    pred_log = {
        "spatial_zero_shot": predict_log(source_model, Xt_te, device),
        "spatial_scratch": predict_log(scratch_model, Xt_te, device),
        "spatial_transfer": predict_log(transfer_model, Xt_te, device),
    }
    pred_mm = {k: invert_log_prediction(v) for k, v in pred_log.items()}
    y_test = np.asarray(a["y_target_test_mm"], dtype=np.float64)
    dates = a["dates_target_test"].astype(str)
    lat_test = np.asarray(a["latitude_target_test"], dtype=float)
    lon_test = np.asarray(a["longitude_target_test"], dtype=float)

    pred_frame = pd.DataFrame({
        "date": dates, "latitude": lat_test, "longitude": lon_test,
        "imd_actual_mm": y_test,
        "spatial_zero_shot_gru_pred_mm": pred_mm["spatial_zero_shot"],
        "spatial_scratch_gru_pred_mm": pred_mm["spatial_scratch"],
        "spatial_transfer_gru_pred_mm": pred_mm["spatial_transfer"],
    })
    prediction_path = PREDICTIONS_DIR / "spatial_gru_test_2023_predictions.csv"
    pred_frame.to_csv(prediction_path, index=False)

    rows = [metric_row(k, y_test, v) for k, v in pred_mm.items()]
    # Simple baselines calculated without using 2023 labels.
    train_y = np.asarray(a["y_target_train_mm"], dtype=float)
    train_lat = np.asarray(a["latitude_target_train"], dtype=float)
    train_lon = np.asarray(a["longitude_target_train"], dtype=float)
    cell_means = pd.DataFrame({"lat": train_lat, "lon": train_lon, "y": train_y}).groupby(["lat", "lon"])["y"].mean()
    climatology_pred = np.array([cell_means.loc[(la, lo)] for la, lo in zip(lat_test, lon_test)], dtype=float)
    rows.append(metric_row("target_train_grid_cell_climatology", y_test, climatology_pred))
    rows.append(metric_row("always_zero", y_test, np.zeros_like(y_test)))
    metrics = pd.DataFrame(rows)
    metrics["evaluation_note"] = "2023 exploratory/post-hoc: this period was inspected in earlier experiments; do not describe this as a fresh untouched test."
    metrics_path = METRICS_DIR / "spatial_gru_metrics_2023_exploratory.csv"
    metrics.to_csv(metrics_path, index=False)

    # Preserve training curves and concise machine-readable manifest.
    history_rows = [r for hist in histories.values() for r in hist]
    pd.DataFrame(history_rows).to_csv(METRICS_DIR / "spatial_gru_training_history.csv", index=False)
    plot_history(histories)
    plot_predictions(y_test, pred_mm)

    manifest = {
        "experiment": "spatially informed single-stage GRU",
        "input_features": ATMOSPHERIC_FEATURES + 2,
        "input_shape_per_sample": [LOOKBACK, INPUT_SIZE],
        "added_features": ["standardized latitude", "standardized longitude"],
        "coordinate_scaler_fit_period": "2014-2017 ERA5 source training samples only",
        "source_training_period": "2014-2017",
        "source_validation_period": "2018",
        "target_training_period": "2019-2021",
        "target_validation_period": "2022",
        "evaluation_period": "2023 exploratory/post-hoc",
        "hidden_size": hidden_size,
        "seed_source": SEED,
        "seed_target": SEED + 1,
        "best_epochs": {"source": source_epoch, "scratch": scratch_epoch, "transfer": transfer_epoch},
        "best_validation_mse_log1p": {"source": source_val_loss, "scratch": scratch_val_loss, "transfer": transfer_val_loss},
        "prediction_file": str(prediction_path),
        "metrics_file": str(metrics_path),
        "warning": "2023 was previously inspected and is therefore exploratory/post-hoc, not a fresh untouched test set.",
    }
    manifest_path = METRICS_DIR / "spatial_gru_experiment_2023_exploratory.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logging.info("SPATIAL GRU EXPERIMENT COMPLETED")
    logging.info("Rows evaluated: %d", len(y_test))
    logging.info("spatial_zero_shot | MAE=%.4f RMSE=%.4f NSE=%.4f KGE=%.4f CSI=%.4f",
                 rows[0]["MAE_mm"], rows[0]["RMSE_mm"], rows[0]["NSE"], rows[0]["KGE"], rows[0]["CSI"])
    logging.info("spatial_scratch | MAE=%.4f RMSE=%.4f NSE=%.4f KGE=%.4f CSI=%.4f",
                 rows[1]["MAE_mm"], rows[1]["RMSE_mm"], rows[1]["NSE"], rows[1]["KGE"], rows[1]["CSI"])
    logging.info("spatial_transfer | MAE=%.4f RMSE=%.4f NSE=%.4f KGE=%.4f CSI=%.4f",
                 rows[2]["MAE_mm"], rows[2]["RMSE_mm"], rows[2]["NSE"], rows[2]["KGE"], rows[2]["CSI"])
    logging.info("Saved predictions: %s", prediction_path)
    logging.info("Saved metrics: %s", metrics_path)
    logging.info("Saved figures: spatial_gru_training_curves.png, spatial_gru_test_scatter_2023.png")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Spatial GRU experiment failed")
        raise
