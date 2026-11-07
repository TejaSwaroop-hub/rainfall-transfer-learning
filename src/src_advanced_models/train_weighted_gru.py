#!/usr/bin/env python3
"""Train intensity-weighted scratch and ERA5-transfer GRUs on IMD rainfall.

This is a separate, controlled experiment. It does not overwrite the original
15-feature GRU or two-stage experiment artifacts. Training samples are weighted
by rainfall-intensity class using target_train labels only. Checkpoint selection
uses ordinary (unweighted) validation MSE in log1p space for comparability.
The previously inspected 2023 period is evaluated for exploratory/post-hoc
comparison only, not as a fresh untouched test.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import random
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
SOURCE_CHECKPOINT = ROOT / "results" / "checkpoints" / "era5_pretrained_gru.pt"
CHECKPOINT_DIR = ROOT / "results" / "checkpoints"
METRICS_DIR = ROOT / "results" / "metrics"
PREDICTIONS_DIR = ROOT / "results" / "predictions"
FIGURES_DIR = ROOT / "results" / "figures"
LOG_PATH = ROOT / "logs" / "intensity_weighted_gru_training.log"

SEED = 242
LOOKBACK = 7
INPUT_SIZE = 15
HIDDEN_SIZE = 64
BATCH_SIZE = 512
MAX_EPOCHS = 40
PATIENCE = 8
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
RAIN_EVENT_THRESHOLD_MM = 2.5
CLASS_EDGES = (0.0, 0.1, 2.5, 15.6, 64.5, float("inf"))
CLASS_NAMES = ("dry_trace_0_0.1", "very_light_0.1_2.5", "moderate_2.5_15.6", "heavy_15.6_64.5", "very_heavy_ge_64.5")
MAX_CLASS_WEIGHT = 3.0
HUBER_BETA = 0.5


class GRURegressor(nn.Module):
    def __init__(self, input_size: int = INPUT_SIZE, hidden_size: int = HIDDEN_SIZE) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.gru(x)
        return self.head(sequence_output[:, -1, :]).squeeze(-1)


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


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"ERA5 pretrained checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # compatibility with older PyTorch versions
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise ValueError("Expected checkpoint dictionary containing model_state_dict.")
    if int(payload.get("input_size", INPUT_SIZE)) != INPUT_SIZE:
        raise ValueError("Pretrained checkpoint input_size is not 15; cannot use this experiment safely.")
    return payload


def make_class_weights(y_train_mm: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Compute capped inverse-square-root class weights from target_train only."""
    y = np.asarray(y_train_mm, dtype=np.float64).reshape(-1)
    if y.size == 0 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Training rainfall labels must be non-empty, finite and non-negative.")
    class_ids = np.digitize(y, CLASS_EDGES[1:-1], right=False)
    counts = np.bincount(class_ids, minlength=len(CLASS_NAMES)).astype(np.int64)
    raw_weights = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    present = counts > 0
    raw_weights[present] = np.sqrt(y.size / (len(CLASS_NAMES) * counts[present]))
    raw_weights = np.clip(raw_weights, 0.0, MAX_CLASS_WEIGHT)
    per_sample = raw_weights[class_ids]
    mean_weight = float(per_sample.mean())
    if mean_weight <= 0 or not np.isfinite(mean_weight):
        raise ValueError("Could not calculate finite class weights.")
    class_weights = np.minimum(raw_weights / mean_weight, MAX_CLASS_WEIGHT)
    per_sample = class_weights[class_ids].astype(np.float32)
    # Capping after normalization can make the average slightly below one;
    # this is intentional so the maximum class weight is never exceeded.

    rows: list[dict[str, Any]] = []
    for i, name in enumerate(CLASS_NAMES):
        low, high = CLASS_EDGES[i], CLASS_EDGES[i + 1]
        rows.append({
            "rainfall_class": name,
            "lower_bound_mm": low,
            "upper_bound_mm": None if not np.isfinite(high) else high,
            "training_samples": int(counts[i]),
            "training_percent": float(100.0 * counts[i] / y.size),
            "sample_weight": float(class_weights[i]),
        })
    return per_sample, rows


def make_loader(X: np.ndarray, y_log: np.ndarray, weights: np.ndarray | None,
                shuffle: bool, seed: int, device: torch.device) -> DataLoader:
    xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    yt = torch.from_numpy(np.ascontiguousarray(y_log, dtype=np.float32).reshape(-1))
    if len(xt) != len(yt):
        raise ValueError("Feature and target sample counts do not match.")
    if weights is None:
        wt = torch.ones(len(yt), dtype=torch.float32)
    else:
        wt = torch.from_numpy(np.ascontiguousarray(weights, dtype=np.float32).reshape(-1))
        if len(wt) != len(yt) or not torch.isfinite(wt).all() or (wt <= 0).any():
            raise ValueError("Sample weights must be finite, positive and match the targets.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(xt, yt, wt), batch_size=BATCH_SIZE, shuffle=shuffle,
        generator=generator if shuffle else None, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


def train_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
                optimizer: torch.optim.Optimizer) -> float:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    huber = nn.SmoothL1Loss(reduction="none", beta=HUBER_BETA)
    for xb, yb, wb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        wb = wb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        per_sample_loss = huber(pred, yb)
        loss = torch.sum(per_sample_loss * wb) / torch.sum(wb)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite intensity-weighted training loss.")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        n = len(xb)
        loss_sum += float(loss.detach().item()) * n
        sample_count += n
    return loss_sum / max(sample_count, 1)


def validate_mse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Unweighted validation objective, comparable with the original GRU runs."""
    model.eval()
    total = 0.0
    count = 0
    mse = nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for xb, yb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            total += float(mse(model(xb), yb).item())
            count += len(xb)
    return total / max(count, 1)


def fit_model(model: GRURegressor, train_loader: DataLoader, val_loader: DataLoader,
              device: torch.device, name: str, checkpoint_path: Path,
              init_type: str) -> tuple[GRURegressor, list[dict[str, float]], int, float]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    waiting = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, device, optimizer)
        val_mse = validate_mse(model, val_loader, device)
        history.append({"model": name, "epoch": float(epoch), "train_weighted_huber": train_loss,
                        "validation_mse_log1p_unweighted": val_mse})
        logging.info("%s | epoch %02d/%02d | train weighted Huber=%.6f | val unweighted MSE(log1p)=%.6f",
                     name, epoch, MAX_EPOCHS, train_loss, val_mse)
        if val_mse < best_val - 1e-7:
            best_val = val_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            waiting = 0
        else:
            waiting += 1
            if waiting >= PATIENCE:
                logging.info("%s | early stopping after epoch %d", name, epoch)
                break
    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {name}.")
    model.load_state_dict(best_state)
    torch.save({
        "model_state_dict": best_state,
        "model_name": name,
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "lookback_days": LOOKBACK,
        "best_epoch": best_epoch,
        "best_validation_mse_log1p_unweighted": best_val,
        "initialization": init_type,
        "training_loss": "class-weighted SmoothL1/Huber on log1p rainfall",
        "validation_selection": "unweighted MSE on log1p rainfall; 2022 only",
        "class_weight_cap": MAX_CLASS_WEIGHT,
        "huber_beta": HUBER_BETA,
        "seed": SEED,
    }, checkpoint_path)
    logging.info("%s | best epoch=%d | best validation MSE=%.6f | saved=%s",
                 name, best_epoch, best_val, checkpoint_path)
    return model, history, best_epoch, best_val


def predict_log(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    tensor_x = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    loader = DataLoader(TensorDataset(tensor_x), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    with torch.no_grad():
        for (xb,) in loader:
            outputs.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def to_mm(pred_log: np.ndarray) -> np.ndarray:
    return np.maximum(np.expm1(np.maximum(np.asarray(pred_log, dtype=np.float64), 0.0)), 0.0)


def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if b != 0 else float("nan")


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if actual.shape != predicted.shape or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metric inputs must be finite arrays with identical shapes.")
    err = predicted - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = float(np.sum((actual - actual.mean()) ** 2))
    nse = float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else float("nan")
    obs_std, pred_std = float(actual.std()), float(predicted.std())
    corr = float(np.corrcoef(actual, predicted)[0, 1]) if obs_std > 0 and pred_std > 0 else float("nan")
    alpha = safe_ratio(pred_std, obs_std)
    beta = safe_ratio(float(predicted.mean()), float(actual.mean()))
    kge = (float(1 - math.sqrt((corr - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))
           if all(np.isfinite([corr, alpha, beta])) else float("nan"))
    obs_event = actual >= RAIN_EVENT_THRESHOLD_MM
    pred_event = predicted >= RAIN_EVENT_THRESHOLD_MM
    tp = int(np.sum(obs_event & pred_event))
    fp = int(np.sum(~obs_event & pred_event))
    fn = int(np.sum(obs_event & ~pred_event))
    tn = int(np.sum(~obs_event & ~pred_event))
    pod = safe_ratio(tp, tp + fn)
    far = safe_ratio(fp, tp + fp)
    csi = safe_ratio(tp, tp + fp + fn)
    hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = safe_ratio(2.0 * (tp * tn - fp * fn), hss_den)
    return {
        "n_samples": int(actual.size), "MAE_mm": mae, "RMSE_mm": rmse, "NSE": nse, "KGE": kge,
        "event_threshold_mm": RAIN_EVENT_THRESHOLD_MM, "POD": pod, "FAR": far, "CSI": csi, "HSS": hss,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def per_class_metrics(actual: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    rows = []
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    for i, name in enumerate(CLASS_NAMES):
        low, high = CLASS_EDGES[i], CLASS_EDGES[i + 1]
        mask = (actual >= low) & (actual < high)
        if not np.any(mask):
            continue
        rows.append({
            "observed_rainfall_class": name,
            "n_samples": int(mask.sum()),
            "actual_mean_mm": float(actual[mask].mean()),
            "predicted_mean_mm": float(predicted[mask].mean()),
            "MAE_mm": float(np.mean(np.abs(predicted[mask] - actual[mask]))),
            "RMSE_mm": float(np.sqrt(np.mean((predicted[mask] - actual[mask]) ** 2))),
        })
    return pd.DataFrame(rows)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def save_plots(histories: dict[str, list[dict[str, float]]], actual: np.ndarray,
               predictions: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, label in (("weighted_scratch", "Weighted scratch GRU"),
                        ("weighted_transfer", "Weighted transfer GRU")):
        hist = histories[name]
        epochs = [x["epoch"] for x in hist]
        axes[0].plot(epochs, [x["train_weighted_huber"] for x in hist], label=f"{label} train")
        axes[1].plot(epochs, [x["validation_mse_log1p_unweighted"] for x in hist], label=f"{label} validation")
    axes[0].set_title("Intensity-weighted training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Weighted Huber loss on log1p rainfall")
    axes[1].set_title("Unweighted validation loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE on log1p rainfall")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "intensity_weighted_gru_training_curves.png", dpi=170)
    plt.close(fig)

    max_obs = float(np.max(actual))
    zoom = min(30.0, max_obs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for key, label in (("weighted_scratch", "Weighted scratch GRU"),
                       ("weighted_transfer", "Weighted transfer GRU")):
        pred = predictions[key]
        axes[0].scatter(actual, pred, s=7, alpha=0.18, label=label)
        mask = actual <= zoom
        axes[1].scatter(actual[mask], pred[mask], s=7, alpha=0.18, label=label)
    for ax, limit, title in ((axes[0], max(max_obs, 1), "Full rainfall range"),
                             (axes[1], max(zoom, 1), f"Observed rainfall ≤ {zoom:g} mm/day")):
        ax.plot([0, limit], [0, limit], "k--", linewidth=1, label="Ideal 1:1")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_xlabel("Observed IMD rainfall (mm/day)")
        ax.set_ylabel("Predicted rainfall (mm/day)")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("Intensity-weighted GRU: 2023 exploratory evaluation")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "intensity_weighted_gru_scatter_2023_exploratory.png", dpi=170)
    plt.close(fig)


def main() -> None:
    configure_logging()
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"Sequence NPZ not found: {NPZ_PATH}")
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Loading prepared sequences: %s", NPZ_PATH)
    logging.info("PyTorch: %s | device: %s", torch.__version__, device)
    if device.type == "cuda":
        logging.info("GPU: %s", torch.cuda.get_device_name(0))

    required = [
        "X_target_train", "y_target_train_mm", "y_target_train_log1p",
        "X_target_val", "y_target_val_mm", "y_target_val_log1p",
        "X_target_test", "y_target_test_mm", "dates_target_test",
        "latitude_target_test", "longitude_target_test",
    ]
    with np.load(NPZ_PATH, allow_pickle=False) as data:
        missing = [k for k in required if k not in data.files]
        if missing:
            raise KeyError(f"Sequence NPZ missing arrays: {missing}")
        arrays = {k: data[k] for k in required}

    X_train = arrays["X_target_train"]
    X_val = arrays["X_target_val"]
    X_test = arrays["X_target_test"]
    y_train_mm = np.asarray(arrays["y_target_train_mm"], dtype=np.float64)
    y_train_log = np.asarray(arrays["y_target_train_log1p"], dtype=np.float32)
    y_val_log = np.asarray(arrays["y_target_val_log1p"], dtype=np.float32)
    y_val_mm = np.asarray(arrays["y_target_val_mm"], dtype=np.float64)
    y_test_mm = np.asarray(arrays["y_target_test_mm"], dtype=np.float64)

    for split, X, y in (("train", X_train, y_train_mm), ("validation", X_val, y_val_mm), ("test", X_test, y_test_mm)):
        if X.ndim != 3 or X.shape[1:] != (LOOKBACK, INPUT_SIZE):
            raise ValueError(f"{split}: expected (N, 7, 15), received {X.shape}")
        if len(X) != len(y) or not np.isfinite(X).all() or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError(f"{split}: malformed or invalid feature/target arrays")
    if not np.allclose(y_train_log, np.log1p(y_train_mm), atol=2e-5):
        raise ValueError("Training mm and log1p targets do not agree; stopping rather than training on inconsistent data.")
    if not np.allclose(y_val_log, np.log1p(y_val_mm), atol=2e-5):
        raise ValueError("Validation mm and log1p targets do not agree.")
    logging.info("Shapes verified: train=%s validation=%s test=%s", X_train.shape, X_val.shape, X_test.shape)
    logging.info("Training period 2019-2021; validation 2022; 2023 evaluation is exploratory/post-hoc.")

    sample_weights, weight_rows = make_class_weights(y_train_mm)
    weight_frame = pd.DataFrame(weight_rows)
    weight_frame.to_csv(METRICS_DIR / "intensity_weighted_gru_class_weights.csv", index=False)
    logging.info("Training rainfall distribution and class weights (computed from target_train only):")
    for row in weight_rows:
        logging.info("%s | n=%d (%.3f%%) | weight=%.4f", row["rainfall_class"], row["training_samples"],
                     row["training_percent"], row["sample_weight"])

    # Create a fresh loader for each model so both runs start from the same
    # deterministic shuffle sequence rather than sharing an advancing generator.
    val_loader = make_loader(X_val, y_val_log, None, False, SEED, device)
    source_payload = load_checkpoint(SOURCE_CHECKPOINT, device)
    source_state = source_payload["model_state_dict"]
    hidden = int(source_payload.get("hidden_size", HIDDEN_SIZE))
    if hidden != HIDDEN_SIZE:
        logging.warning("Checkpoint hidden size=%d differs from default=%d; using checkpoint size.", hidden, HIDDEN_SIZE)

    histories: dict[str, list[dict[str, float]]] = {}
    trained: dict[str, GRURegressor] = {}
    selections: dict[str, dict[str, Any]] = {}

    # Scratch model. Use a fixed but separate seed for this experiment.
    set_seed(SEED + 1)
    scratch = GRURegressor(hidden_size=hidden)
    scratch_train_loader = make_loader(X_train, y_train_log, sample_weights, True, SEED, device)
    scratch, hist, epoch, val = fit_model(
        scratch, scratch_train_loader, val_loader, device, "Intensity-weighted scratch GRU",
        CHECKPOINT_DIR / "intensity_weighted_scratch_gru.pt", "random initialization")
    trained["weighted_scratch"] = scratch
    histories["weighted_scratch"] = hist
    selections["weighted_scratch"] = {"best_epoch": epoch, "best_validation_mse_log1p_unweighted": val}

    # Transfer model: match the original 15-feature architecture, copy only its GRU encoder.
    set_seed(SEED + 1)
    transfer = GRURegressor(hidden_size=hidden)
    encoder_state = {k.removeprefix("gru."): v for k, v in source_state.items() if k.startswith("gru.")}
    if not encoder_state:
        raise ValueError("No GRU encoder weights found in the ERA5 pretrained checkpoint.")
    transfer.gru.load_state_dict(copy.deepcopy(encoder_state), strict=True)
    transfer_train_loader = make_loader(X_train, y_train_log, sample_weights, True, SEED, device)
    transfer, hist, epoch, val = fit_model(
        transfer, transfer_train_loader, val_loader, device, "Intensity-weighted transfer GRU",
        CHECKPOINT_DIR / "intensity_weighted_transfer_gru.pt", "ERA5-pretrained GRU encoder + new regression head")
    trained["weighted_transfer"] = transfer
    histories["weighted_transfer"] = hist
    selections["weighted_transfer"] = {"best_epoch": epoch, "best_validation_mse_log1p_unweighted": val}

    histories_frame = pd.DataFrame([row for hist in histories.values() for row in hist])
    histories_frame.to_csv(METRICS_DIR / "intensity_weighted_gru_training_history.csv", index=False)

    predictions: dict[str, np.ndarray] = {}
    for key, model in trained.items():
        predictions[key] = to_mm(predict_log(model, X_test, device))
    metrics_rows = []
    for key, pred in predictions.items():
        metrics_rows.append({"model": key, **calculate_metrics(y_test_mm, pred),
                             "evaluation_note": "2023 exploratory/post-hoc; period inspected in earlier experiments"})
    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(METRICS_DIR / "intensity_weighted_gru_metrics_2023_exploratory.csv", index=False)

    prediction_frame = pd.DataFrame({
        "date": arrays["dates_target_test"].astype(str),
        "latitude": arrays["latitude_target_test"].astype(float),
        "longitude": arrays["longitude_target_test"].astype(float),
        "imd_actual_mm": y_test_mm,
        "intensity_weighted_scratch_gru_pred_mm": predictions["weighted_scratch"],
        "intensity_weighted_transfer_gru_pred_mm": predictions["weighted_transfer"],
    })
    prediction_frame.to_csv(PREDICTIONS_DIR / "intensity_weighted_gru_test_2023_predictions.csv", index=False)

    # Combine for comparison without modifying the existing model-comparison files.
    base_comparison = METRICS_DIR / "model_comparison_all_models_test_2023.csv"
    if base_comparison.exists():
        comparison = pd.read_csv(base_comparison)
        comparison = pd.concat([comparison, metrics_frame], ignore_index=True, sort=False)
    else:
        comparison = metrics_frame.copy()
    comparison.to_csv(METRICS_DIR / "model_comparison_with_intensity_weighted_2023_exploratory.csv", index=False)

    class_error = pd.concat([
        per_class_metrics(y_test_mm, predictions["weighted_scratch"]).assign(model="intensity_weighted_scratch"),
        per_class_metrics(y_test_mm, predictions["weighted_transfer"]).assign(model="intensity_weighted_transfer"),
    ], ignore_index=True)
    class_error.to_csv(METRICS_DIR / "intensity_weighted_gru_error_by_class_2023.csv", index=False)

    save_plots(histories, y_test_mm, predictions)
    payload = {
        "experiment": "intensity-weighted regression GRU",
        "source_encoder_checkpoint": str(SOURCE_CHECKPOINT),
        "target_training_period": "2019-2021",
        "target_validation_period": "2022",
        "evaluation_period": "2023 exploratory/post-hoc; not a fresh untouched test",
        "input_shape": [LOOKBACK, INPUT_SIZE],
        "training_loss": "class-weighted SmoothL1/Huber on log1p rainfall",
        "class_edges_mm": list(CLASS_EDGES[:-1]) + [None],
        "class_weight_cap": MAX_CLASS_WEIGHT,
        "huber_beta": HUBER_BETA,
        "class_weights_from_target_train_only": weight_rows,
        "checkpoint_selection": "unweighted validation MSE on log1p rainfall, 2022 only",
        "seed": SEED,
        "models": selections,
        "metrics_2023_exploratory": metrics_rows,
        "caveats": [
            "Weights were computed exclusively from 2019-2021 IMD training labels.",
            "Validation checkpoint selection used unweighted 2022 MSE in log1p space.",
            "The 2023 period had previously been inspected; these numbers are exploratory/post-hoc.",
            "Only 81 target-training observations were in the >=64.5 mm/day class; capped weights do not create additional extreme events.",
            "Intensity weighting may improve rare-event fit while worsening aggregate errors; all metrics must be considered.",
        ],
    }
    (METRICS_DIR / "intensity_weighted_gru_experiment.json").write_text(
        json.dumps(json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")

    logging.info("INTENSITY-WEIGHTED GRU EXPERIMENT COMPLETED")
    for row in metrics_rows:
        logging.info("%s | MAE=%.4f | RMSE=%.4f | NSE=%.4f | KGE=%s | POD=%.4f | FAR=%s | CSI=%.4f",
                     row["model"], row["MAE_mm"], row["RMSE_mm"], row["NSE"],
                     "N/A" if not np.isfinite(row["KGE"]) else f"{row['KGE']:.4f}", row["POD"],
                     "N/A" if not np.isfinite(row["FAR"]) else f"{row['FAR']:.4f}", row["CSI"])
    logging.info("Saved checkpoints: intensity_weighted_scratch_gru.pt, intensity_weighted_transfer_gru.pt")
    logging.info("Saved metrics: intensity_weighted_gru_metrics_2023_exploratory.csv")
    logging.info("Saved combined comparison: model_comparison_with_intensity_weighted_2023_exploratory.csv")
    logging.info("Saved predictions: intensity_weighted_gru_test_2023_predictions.csv")
    logging.info("Saved figures: intensity_weighted_gru_training_curves.png, intensity_weighted_gru_scatter_2023_exploratory.png")


if __name__ == "__main__":
    main()
