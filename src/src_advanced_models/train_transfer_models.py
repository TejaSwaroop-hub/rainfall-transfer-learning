#!/usr/bin/env python3
"""Train ERA5-pretrained GRU, scratch GRU, and evaluate on IMD 2023.

All model selection uses the designated validation splits. The final 2023 test
set is used only after checkpoints have been selected.
"""
from __future__ import annotations

import copy
import csv
import json
import logging
import math
import random
from pathlib import Path

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
CHECKPOINT_DIR = ROOT / "results" / "checkpoints"
METRICS_DIR = ROOT / "results" / "metrics"
PREDICTIONS_DIR = ROOT / "results" / "predictions"
FIGURES_DIR = ROOT / "results" / "figures"
LOG_PATH = ROOT / "logs" / "model_training.log"

SEED = 42
BATCH_SIZE = 512
HIDDEN_SIZE = 64
MAX_SOURCE_EPOCHS = 35
MAX_TARGET_EPOCHS = 40
SOURCE_PATIENCE = 6
TARGET_PATIENCE = 8
SOURCE_LR = 1e-3
TARGET_LR = 5e-4
WEIGHT_DECAY = 1e-5
RAIN_THRESHOLD_MM = 2.5


class GRURegressor(nn.Module):
    def __init__(self, input_size: int = 15, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
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
        final_state = sequence_output[:, -1, :]
        return self.head(final_state).squeeze(-1)


def setup_logging() -> None:
    for directory in (CHECKPOINT_DIR, METRICS_DIR, PREDICTIONS_DIR, FIGURES_DIR, LOG_PATH.parent):
        directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH, mode="w")],
        force=True,
    )


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool, device: torch.device) -> DataLoader:
    tensor_x = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    tensor_y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return DataLoader(
        TensorDataset(tensor_x, tensor_y),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        generator=generator if shuffle else None,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    with torch.set_grad_enabled(training):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            count = batch_x.shape[0]
            total_loss += float(loss.detach().item()) * count
            total_count += count
    return total_loss / max(total_count, 1)


def fit_with_early_stopping(
    model: GRURegressor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    checkpoint_path: Path,
    model_name: str,
) -> tuple[GRURegressor, list[dict[str, float]], int, float]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss = run_epoch(model, val_loader, loss_fn, device)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        logging.info(
            "%s | epoch %02d/%02d | train MSE(log1p)=%.6f | val MSE(log1p)=%.6f",
            model_name, epoch, max_epochs, train_loss, val_loss,
        )
        if epochs_without_improvement >= patience:
            logging.info("%s | early stopping after %d epochs.", model_name, epoch)
            break

    if best_state is None:
        raise RuntimeError(f"No valid checkpoint was produced for {model_name}.")

    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": best_state,
            "model_name": model_name,
            "input_size": 15,
            "hidden_size": HIDDEN_SIZE,
            "lookback_days": 7,
            "best_epoch": best_epoch,
            "best_validation_mse_log1p": best_val,
            "seed": SEED,
        },
        checkpoint_path,
    )
    logging.info(
        "%s | best epoch=%d | best validation MSE(log1p)=%.6f | saved=%s",
        model_name, best_epoch, best_val, checkpoint_path,
    )
    return model, history, best_epoch, best_val


def predict_log(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    tensor_x = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    loader = DataLoader(TensorDataset(tensor_x), batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            outputs.append(model(batch_x).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def log_to_mm(pred_log: np.ndarray) -> np.ndarray:
    # Rainfall cannot be negative; clip in transformed space before expm1.
    return np.maximum(np.expm1(np.maximum(pred_log, 0.0)), 0.0)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator != 0 else float("nan")


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = predicted - actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    nse = float(1.0 - np.sum(error ** 2) / denominator) if denominator > 0 else float("nan")

    actual_std = float(np.std(actual))
    predicted_std = float(np.std(predicted))
    if actual_std > 0 and predicted_std > 0:
        correlation = float(np.corrcoef(actual, predicted)[0, 1])
    else:
        correlation = float("nan")
    beta = safe_ratio(float(np.mean(predicted)), float(np.mean(actual)))
    alpha = safe_ratio(predicted_std, actual_std)
    if all(np.isfinite([correlation, alpha, beta])):
        kge = float(1.0 - math.sqrt((correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    else:
        kge = float("nan")

    observed_event = actual >= RAIN_THRESHOLD_MM
    predicted_event = predicted >= RAIN_THRESHOLD_MM
    tp = int(np.sum(observed_event & predicted_event))
    fp = int(np.sum(~observed_event & predicted_event))
    fn = int(np.sum(observed_event & ~predicted_event))
    tn = int(np.sum(~observed_event & ~predicted_event))
    pod = safe_ratio(tp, tp + fn)
    far = safe_ratio(fp, tp + fp)
    csi = safe_ratio(tp, tp + fp + fn)
    hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = safe_ratio(2.0 * (tp * tn - fp * fn), hss_den)

    return {
        "n_samples": int(actual.size),
        "MAE_mm": mae,
        "RMSE_mm": rmse,
        "NSE": nse,
        "KGE": kge,
        "event_threshold_mm": RAIN_THRESHOLD_MM,
        "POD": pod,
        "FAR": far,
        "CSI": csi,
        "HSS": hss,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


def json_safe(value):
    """Convert NumPy values and non-finite floats to JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def grid_cell_climatology(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    y_train = np.asarray(npz["y_target_train_mm"], dtype=np.float64)
    lat_train = np.asarray(npz["latitude_target_train"], dtype=np.float64)
    lon_train = np.asarray(npz["longitude_target_train"], dtype=np.float64)
    lat_test = np.asarray(npz["latitude_target_test"], dtype=np.float64)
    lon_test = np.asarray(npz["longitude_target_test"], dtype=np.float64)
    keys_train = np.round(lat_train, 5).astype(str) + "_" + np.round(lon_train, 5).astype(str)
    keys_test = np.round(lat_test, 5).astype(str) + "_" + np.round(lon_test, 5).astype(str)
    means = {key: float(np.mean(y_train[keys_train == key])) for key in np.unique(keys_train)}
    missing = sorted(set(keys_test) - set(means))
    if missing:
        raise ValueError(f"No training climatology for test grid cells: {missing}")
    return np.asarray([means[key] for key in keys_test], dtype=np.float64)


def save_history_csv(histories: dict[str, list[dict[str, float]]]) -> None:
    rows = []
    for model_name, history in histories.items():
        for row in history:
            rows.append({"model": model_name, **row})
    pd.DataFrame(rows).to_csv(METRICS_DIR / "training_history.csv", index=False)


def save_training_plot(histories: dict[str, list[dict[str, float]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    source = histories["era5_pretrained_gru"]
    axes[0].plot([r["epoch"] for r in source], [r["train_loss"] for r in source], label="Train")
    axes[0].plot([r["epoch"] for r in source], [r["val_loss"] for r in source], label="Validation")
    axes[0].set_title("ERA5 pretraining")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE on log1p rainfall")
    axes[0].legend()
    for model_name, label in (("scratch_gru", "Scratch GRU"), ("transfer_gru", "Transfer GRU")):
        history = histories[model_name]
        axes[1].plot([r["epoch"] for r in history], [r["val_loss"] for r in history], label=label)
    axes[1].set_title("IMD fine-tuning / training")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation MSE on log1p rainfall")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "gru_training_curves.png", dpi=160)
    plt.close(fig)


def main() -> None:
    setup_logging()
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"Sequence file not found: {NPZ_PATH}")

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Loading prepared sequences: %s", NPZ_PATH)
    logging.info("PyTorch version: %s", torch.__version__)
    logging.info("Training device: %s", device)
    if device.type == "cuda":
        logging.info("GPU: %s", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()
    else:
        logging.warning("CUDA is unavailable; training will use CPU.")

    with np.load(NPZ_PATH, allow_pickle=False) as data:
        required = [
            "X_source_train", "y_source_train_log1p", "X_source_val", "y_source_val_log1p",
            "X_target_train", "y_target_train_log1p", "X_target_val", "y_target_val_log1p",
            "X_target_test", "y_target_test_mm", "dates_target_test",
            "latitude_target_train", "longitude_target_train",
            "latitude_target_test", "longitude_target_test",
        ]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"Sequence NPZ missing required arrays: {missing}")

        arrays = {key: data[key] for key in required}

    for split in ("source_train", "source_val", "target_train", "target_val", "target_test"):
        X = arrays[f"X_{split}"]
        if X.ndim != 3 or X.shape[1:] != (7, 15):
            raise ValueError(f"Unexpected X_{split} shape: {X.shape}; expected (N, 7, 15).")
        if not np.isfinite(X).all():
            raise ValueError(f"X_{split} contains non-finite values.")
    logging.info("Input array shapes verified: 7 days x 15 features per sample.")

    # Build loaders. Validation loaders never shuffle.
    source_train_loader = make_loader(arrays["X_source_train"], arrays["y_source_train_log1p"], True, device)
    source_val_loader = make_loader(arrays["X_source_val"], arrays["y_source_val_log1p"], False, device)
    target_val_loader = make_loader(arrays["X_target_val"], arrays["y_target_val_log1p"], False, device)

    histories: dict[str, list[dict[str, float]]] = {}

    # 1) ERA5 source pretraining.
    set_seed(SEED)
    source_model = GRURegressor()
    source_model, histories["era5_pretrained_gru"], source_best_epoch, source_best_val = fit_with_early_stopping(
        source_model, source_train_loader, source_val_loader, device,
        SOURCE_LR, MAX_SOURCE_EPOCHS, SOURCE_PATIENCE,
        CHECKPOINT_DIR / "era5_pretrained_gru.pt", "ERA5-pretrained GRU",
    )

    # 2) Scratch target GRU. Same architecture and target training protocol.
    set_seed(SEED + 1)
    scratch_model = GRURegressor()
    scratch_train_loader = make_loader(arrays["X_target_train"], arrays["y_target_train_log1p"], True, device)
    scratch_model, histories["scratch_gru"], scratch_best_epoch, scratch_best_val = fit_with_early_stopping(
        scratch_model, scratch_train_loader, target_val_loader, device,
        TARGET_LR, MAX_TARGET_EPOCHS, TARGET_PATIENCE,
        CHECKPOINT_DIR / "scratch_gru_imd.pt", "Scratch IMD GRU",
    )

    # 3) Transfer GRU. Copy only the pretrained temporal encoder; initialize a
    # fresh output head because the source and target rainfall products differ.
    set_seed(SEED + 1)
    transfer_model = GRURegressor()
    transfer_model.gru.load_state_dict(copy.deepcopy(source_model.gru.state_dict()))
    transfer_train_loader = make_loader(arrays["X_target_train"], arrays["y_target_train_log1p"], True, device)
    transfer_model, histories["transfer_gru"], transfer_best_epoch, transfer_best_val = fit_with_early_stopping(
        transfer_model, transfer_train_loader, target_val_loader, device,
        TARGET_LR, MAX_TARGET_EPOCHS, TARGET_PATIENCE,
        CHECKPOINT_DIR / "transfer_gru_imd.pt", "ERA5-to-IMD transfer GRU",
    )

    save_history_csv(histories)
    save_training_plot(histories)

    # Final 2023 evaluation is performed only after validation-based checkpoint selection.
    X_test = arrays["X_target_test"]
    y_test = np.asarray(arrays["y_target_test_mm"], dtype=np.float64)
    zero_shot_pred = log_to_mm(predict_log(source_model, X_test, device))
    scratch_pred = log_to_mm(predict_log(scratch_model, X_test, device))
    transfer_pred = log_to_mm(predict_log(transfer_model, X_test, device))
    climatology_pred = grid_cell_climatology(np.load(NPZ_PATH, allow_pickle=False))

    metrics = {
        "spatial_grid_cell_climatology": calculate_metrics(y_test, climatology_pred),
        "zero_shot_era5_gru": calculate_metrics(y_test, zero_shot_pred),
        "scratch_gru": calculate_metrics(y_test, scratch_pred),
        "transfer_gru": calculate_metrics(y_test, transfer_pred),
    }
    prediction_frame = pd.DataFrame({
        "date": arrays["dates_target_test"].astype(str),
        "latitude": arrays["latitude_target_test"].astype(float),
        "longitude": arrays["longitude_target_test"].astype(float),
        "imd_actual_mm": y_test,
        "climatology_pred_mm": climatology_pred,
        "zero_shot_era5_gru_pred_mm": zero_shot_pred,
        "scratch_gru_pred_mm": scratch_pred,
        "transfer_gru_pred_mm": transfer_pred,
    })
    prediction_path = PREDICTIONS_DIR / "imd_test_2023_predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False)

    metrics_payload = {
        "test_period": "2023-01-01 to 2023-12-31",
        "test_samples": int(y_test.size),
        "rain_event_threshold_mm": RAIN_THRESHOLD_MM,
        "selection_rule": "Best checkpoint selected using 2022 validation MSE in log1p space; 2023 used only for final evaluation.",
        "pretraining": {
            "period": "2014-2017 ERA5",
            "validation_period": "2018 ERA5",
            "best_epoch": source_best_epoch,
            "best_validation_mse_log1p": source_best_val,
        },
        "scratch_gru": {"best_epoch": scratch_best_epoch, "best_validation_mse_log1p": scratch_best_val},
        "transfer_gru": {"best_epoch": transfer_best_epoch, "best_validation_mse_log1p": transfer_best_val},
        "metrics": metrics,
        "notes": [
            "All reported test metrics were calculated from model predictions and IMD observations for 2023.",
            "Zero-shot GRU applies the ERA5-pretrained model directly to ERA5 input features and compares its rainfall estimates with IMD observations without target fine-tuning.",
            "Transfer GRU initializes its GRU encoder from ERA5 pretraining and uses a newly initialized regression head before IMD fine-tuning.",
            "Rainfall predictions are clipped at zero after inverse log1p transformation.",
            "Only one final test year is available; interannual generalization claims must be cautious.",
            "ERA5 and IMD rainfall are different products; source-target distribution differences are retained rather than artificially corrected.",
        ],
    }
    metrics_path = METRICS_DIR / "model_comparison_test_2023.json"
    metrics_path.write_text(json.dumps(json_safe(metrics_payload), indent=2, allow_nan=False), encoding="utf-8")

    # Concise CSV comparison for reports.
    comparison_rows = []
    for name, values in metrics.items():
        comparison_rows.append({"model": name, **values})
    pd.DataFrame(comparison_rows).to_csv(METRICS_DIR / "model_comparison_test_2023.csv", index=False)

    # Scatter and error comparison figures.
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sample_n = min(5000, y_test.size)
    rng = np.random.default_rng(SEED)
    selected = rng.choice(y_test.size, size=sample_n, replace=False)
    ax.scatter(y_test[selected], transfer_pred[selected], s=8, alpha=0.35)
    max_value = float(max(np.max(y_test), np.max(transfer_pred)))
    ax.plot([0, max_value], [0, max_value], linestyle="--")
    ax.set_xlabel("Observed IMD rainfall (mm/day)")
    ax.set_ylabel("Transfer GRU prediction (mm/day)")
    ax.set_title("2023 test: transfer GRU predictions")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "transfer_gru_test_scatter_2023.png", dpi=160)
    plt.close(fig)

    logging.info("TRAINING AND FINAL TEST EVALUATION COMPLETED.")
    logging.info("Test samples: %d", y_test.size)
    for name, values in metrics.items():
        logging.info(
            "%s | MAE=%.4f | RMSE=%.4f | NSE=%s | KGE=%s | CSI=%s",
            name, values["MAE_mm"], values["RMSE_mm"],
            f"{values['NSE']:.4f}" if np.isfinite(values["NSE"]) else "undefined",
            f"{values['KGE']:.4f}" if np.isfinite(values["KGE"]) else "undefined",
            f"{values['CSI']:.4f}" if np.isfinite(values["CSI"]) else "undefined",
        )
    logging.info("Saved predictions: %s", prediction_path)
    logging.info("Saved metrics JSON: %s", metrics_path)
    logging.info("Saved metrics CSV: %s", METRICS_DIR / "model_comparison_test_2023.csv")
    logging.info("Saved figures: %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
