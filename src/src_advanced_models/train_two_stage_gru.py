#!/usr/bin/env python3
"""Train two-stage rainfall occurrence/amount GRUs for Anantapur.

This is a separate follow-up experiment. It does not overwrite the original
single-stage GRU checkpoints, predictions, or metrics.

Protocol:
- Fit on IMD target_train (2019-2021).
- Select network checkpoints and event thresholds using target_val (2022).
- Evaluate on target_test (2023), which has already been inspected in the
  preceding experiment; therefore these 2023 results are exploratory/post-hoc,
  not a newly untouched blind test.
- Compare scratch initialization against an encoder transferred from the
  ERA5-pretrained GRU.

Two stages:
1) GRU classifier: predict whether IMD rainfall >= 2.5 mm/day.
2) GRU conditional regressor: estimate log1p rainfall amount on training
   samples whose observed rainfall >= 2.5 mm/day.
The final prediction is zero when the classifier is below its 2022-selected
probability threshold; otherwise it is the conditional amount estimate.
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
LOG_PATH = ROOT / "logs" / "two_stage_gru_training.log"

SEED = 2026
INPUT_SIZE = 15
HIDDEN_SIZE = 64
LOOKBACK = 7
BATCH_SIZE = 512
RAIN_THRESHOLD_MM = 2.5
MAX_CLASS_EPOCHS = 35
MAX_AMOUNT_EPOCHS = 45
CLASS_PATIENCE = 7
AMOUNT_PATIENCE = 9
CLASS_LR = 5e-4
AMOUNT_LR = 4e-4
WEIGHT_DECAY = 1e-5

# Thresholds are selected on 2022 validation data only.
PROBABILITY_THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


class GRUHead(nn.Module):
    """Same 1-layer GRU encoder and 64->32->1 head as the original model."""
    def __init__(self, output_type: str):
        super().__init__()
        if output_type not in {"classifier", "regressor"}:
            raise ValueError("output_type must be 'classifier' or 'regressor'.")
        self.output_type = output_type
        self.gru = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 32),
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def event_counts(actual: np.ndarray, predicted: np.ndarray) -> dict[str, int | float]:
    obs = np.asarray(actual) >= RAIN_THRESHOLD_MM
    pred = np.asarray(predicted) >= RAIN_THRESHOLD_MM
    tp = int(np.sum(obs & pred))
    fp = int(np.sum(~obs & pred))
    fn = int(np.sum(obs & ~pred))
    tn = int(np.sum(~obs & ~pred))
    pod = safe_div(tp, tp + fn)
    far = safe_div(fp, tp + fp)
    csi = safe_div(tp, tp + fp + fn)
    hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = safe_div(2.0 * (tp * tn - fp * fn), hss_den)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "POD": pod, "FAR": far, "CSI": csi, "HSS": hss}


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    err = predicted - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = float(np.sum((actual - np.mean(actual)) ** 2))
    nse = float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else float("nan")
    obs_std = float(np.std(actual))
    pred_std = float(np.std(predicted))
    if obs_std > 0 and pred_std > 0:
        r = float(np.corrcoef(actual, predicted)[0, 1])
        alpha = pred_std / obs_std
        beta = float(np.mean(predicted) / np.mean(actual)) if np.mean(actual) != 0 else float("nan")
        kge = 1.0 - math.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2) if np.isfinite(beta) else float("nan")
    else:
        r = alpha = beta = kge = float("nan")
    return {"n_samples": int(actual.size), "MAE_mm": mae, "RMSE_mm": rmse,
            "NSE": nse, "KGE": kge, **event_counts(actual, predicted)}


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool, device: torch.device, seed: int) -> DataLoader:
    tx = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    ty = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(tx, ty), batch_size=BATCH_SIZE, shuffle=shuffle,
        num_workers=0, pin_memory=(device.type == "cuda"),
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
    total_n = 0
    with torch.set_grad_enabled(training):
        for bx, by in loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(bx)
            loss = loss_fn(output, by)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            n = int(bx.shape[0])
            total_loss += float(loss.detach().item()) * n
            total_n += n
    return total_loss / max(total_n, 1)


def train_model(
    model: GRUHead,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    lr: float,
    max_epochs: int,
    patience: int,
    checkpoint_path: Path,
    model_label: str,
    transfer: bool,
) -> tuple[GRUHead, list[dict[str, float]], int, float]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss = run_epoch(model, val_loader, loss_fn, device)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        logging.info("%s | epoch %02d/%02d | train loss=%.6f | val loss=%.6f",
                     model_label, epoch, max_epochs, train_loss, val_loss)
        if stale >= patience:
            logging.info("%s | early stopping after %d epochs.", model_label, epoch)
            break

    if best_state is None:
        raise RuntimeError(f"No best checkpoint selected for {model_label}.")
    model.load_state_dict(best_state)
    torch.save({
        "model_state_dict": best_state,
        "model_label": model_label,
        "output_type": model.output_type,
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "lookback_days": LOOKBACK,
        "rain_threshold_mm": RAIN_THRESHOLD_MM,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "seed": SEED,
        "transfer_initialized": transfer,
    }, checkpoint_path)
    logging.info("%s | best epoch=%d | best validation loss=%.6f | saved=%s",
                 model_label, best_epoch, best_loss, checkpoint_path)
    return model, history, best_epoch, best_loss


def predict_raw(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    tx = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    loader = DataLoader(TensorDataset(tx), batch_size=BATCH_SIZE, shuffle=False)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for (bx,) in loader:
            outputs.append(model(bx.to(device)).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def initialize_from_pretrained(model: GRUHead, checkpoint: dict[str, Any]) -> None:
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("ERA5 checkpoint does not contain model_state_dict.")
    encoder_state = {key.removeprefix("gru."): value for key, value in state.items() if key.startswith("gru.")}
    if not encoder_state:
        raise ValueError("No GRU encoder weights found in ERA5 checkpoint.")
    model.gru.load_state_dict(encoder_state, strict=True)


def select_threshold(actual_val: np.ndarray, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    actual_event = np.asarray(actual_val) >= RAIN_THRESHOLD_MM
    for threshold in PROBABILITY_THRESHOLDS:
        predicted_event = probabilities >= threshold
        tp = int(np.sum(actual_event & predicted_event))
        fp = int(np.sum(~actual_event & predicted_event))
        fn = int(np.sum(actual_event & ~predicted_event))
        tn = int(np.sum(~actual_event & ~predicted_event))
        csi = safe_div(tp, tp + fp + fn)
        pod = safe_div(tp, tp + fn)
        far = safe_div(fp, tp + fp)
        hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
        hss = safe_div(2.0 * (tp * tn - fp * fn), hss_den)
        rows.append({"threshold": float(threshold), "CSI": csi, "POD": pod, "FAR": far,
                     "HSS": hss, "TP": tp, "FP": fp, "FN": fn, "TN": tn})
    frame = pd.DataFrame(rows)
    # Primary objective: highest validation CSI. Tie-break: highest HSS,
    # then lower FAR, then higher threshold (more conservative event declaration).
    ranked = frame.sort_values(
        ["CSI", "HSS", "FAR", "threshold"],
        ascending=[False, False, True, False],
        na_position="last",
    )
    threshold = float(ranked.iloc[0]["threshold"])
    return threshold, frame


def gated_prediction(probabilities: np.ndarray, conditional_amount: np.ndarray, threshold: float) -> np.ndarray:
    event = np.asarray(probabilities) >= threshold
    amount = np.maximum(np.asarray(conditional_amount, dtype=np.float64), RAIN_THRESHOLD_MM)
    return np.where(event, amount, 0.0)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    return value


def load_npz() -> dict[str, np.ndarray]:
    required = [
        "X_target_train", "y_target_train_mm",
        "X_target_val", "y_target_val_mm",
        "X_target_test", "y_target_test_mm", "dates_target_test",
        "latitude_target_test", "longitude_target_test",
    ]
    with np.load(NPZ_PATH, allow_pickle=False) as data:
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"Sequence NPZ is missing arrays: {missing}")
        arrays = {key: data[key] for key in required}
    for split in ("train", "val", "test"):
        X = arrays[f"X_target_{split}"]
        y = arrays[f"y_target_{split}_mm"]
        if X.ndim != 3 or X.shape[1:] != (LOOKBACK, INPUT_SIZE):
            raise ValueError(f"Unexpected target {split} input shape: {X.shape}")
        if len(X) != len(y) or not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError(f"Invalid or non-finite target {split} arrays.")
        if np.any(y < 0):
            raise ValueError(f"Negative rainfall found in target {split}.")
    return arrays


def save_histories(histories: dict[str, list[dict[str, float]]]) -> None:
    rows = []
    for name, history in histories.items():
        for row in history:
            rows.append({"model": name, **row})
    pd.DataFrame(rows).to_csv(METRICS_DIR / "two_stage_training_history.csv", index=False)


def plot_histories(histories: dict[str, list[dict[str, float]]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, name, title in [
        (axes[0, 0], "scratch_classifier", "Scratch GRU classifier"),
        (axes[0, 1], "transfer_classifier", "Transfer GRU classifier"),
        (axes[1, 0], "scratch_amount", "Scratch GRU amount regressor"),
        (axes[1, 1], "transfer_amount", "Transfer GRU amount regressor"),
    ]:
        hist = histories[name]
        ep = [r["epoch"] for r in hist]
        ax.plot(ep, [r["train_loss"] for r in hist], label="Train")
        ax.plot(ep, [r["val_loss"] for r in hist], label="Validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weighted BCE" if "classifier" in name else "Huber loss on log1p rainfall")
        ax.legend()
    fig.suptitle("Two-stage GRU training and validation curves")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "two_stage_gru_training_curves.png", dpi=170)
    plt.close(fig)


def main() -> None:
    setup_logging()
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"Sequence dataset not found: {NPZ_PATH}")
    if not SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError(f"ERA5 pretrained checkpoint not found: {SOURCE_CHECKPOINT}")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Loading sequence data: %s", NPZ_PATH)
    logging.info("Device: %s", device)
    if device.type == "cuda":
        logging.info("GPU: %s", torch.cuda.get_device_name(0))

    arrays = load_npz()
    X_train = arrays["X_target_train"]
    y_train = arrays["y_target_train_mm"].astype(np.float64)
    X_val = arrays["X_target_val"]
    y_val = arrays["y_target_val_mm"].astype(np.float64)
    X_test = arrays["X_target_test"]
    y_test = arrays["y_target_test_mm"].astype(np.float64)

    y_train_event = (y_train >= RAIN_THRESHOLD_MM).astype(np.float32)
    y_val_event = (y_val >= RAIN_THRESHOLD_MM).astype(np.float32)
    positive_train = y_train >= RAIN_THRESHOLD_MM
    positive_val = y_val >= RAIN_THRESHOLD_MM
    if positive_train.sum() < 100 or positive_val.sum() < 20:
        raise ValueError("Too few rainfall-event samples for a reliable two-stage experiment.")

    n_pos = int(positive_train.sum())
    n_neg = int((~positive_train).sum())
    pos_weight_value = float(n_neg / max(n_pos, 1))
    pos_weight_value = min(pos_weight_value, 20.0)
    logging.info("Target training samples: %d; rainfall events >= %.1f mm: %d (%.2f%%)",
                 len(y_train), RAIN_THRESHOLD_MM, n_pos, 100.0 * n_pos / len(y_train))
    logging.info("Classifier positive-class weight: %.4f", pos_weight_value)
    logging.info("Target validation rainfall events: %d/%d (%.2f%%)",
                 int(positive_val.sum()), len(y_val), 100.0 * positive_val.mean())
    logging.info("Target test rainfall events: %d/%d (%.2f%%); 2023 was previously inspected, so this is exploratory post-hoc evaluation.",
                 int((y_test >= RAIN_THRESHOLD_MM).sum()), len(y_test), 100.0 * np.mean(y_test >= RAIN_THRESHOLD_MM))

    # Weighted BCE is calculated from training data only.
    cls_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))

    # Conditional amount regressors see only observed rainfall events >= threshold.
    X_train_wet = X_train[positive_train]
    y_train_wet_log = np.log1p(y_train[positive_train]).astype(np.float32)
    X_val_wet = X_val[positive_val]
    y_val_wet_log = np.log1p(y_val[positive_val]).astype(np.float32)
    amount_loss = nn.SmoothL1Loss(beta=1.0)

    source_checkpoint = torch.load(SOURCE_CHECKPOINT, map_location="cpu")
    if source_checkpoint.get("hidden_size") != HIDDEN_SIZE or source_checkpoint.get("input_size") != INPUT_SIZE:
        raise ValueError("ERA5 checkpoint architecture does not match this two-stage model.")

    histories: dict[str, list[dict[str, float]]] = {}
    fitted: dict[str, dict[str, Any]] = {}

    # Train scratch and transfer versions of each stage with matching heads/seeds.
    for mode, transfer in (("scratch", False), ("transfer", True)):
        # Recreate loaders so scratch and transfer runs see the same batch order.
        cls_train_loader = make_loader(X_train, y_train_event, True, device, SEED + 1)
        cls_val_loader = make_loader(X_val, y_val_event, False, device, SEED + 2)
        amount_train_loader = make_loader(X_train_wet, y_train_wet_log, True, device, SEED + 3)
        amount_val_loader = make_loader(X_val_wet, y_val_wet_log, False, device, SEED + 4)

        # Classifier
        set_seed(SEED + 10)
        classifier = GRUHead("classifier")
        if transfer:
            initialize_from_pretrained(classifier, source_checkpoint)
        cls_path = CHECKPOINT_DIR / f"two_stage_{mode}_classifier.pt"
        classifier, histories[f"{mode}_classifier"], cls_epoch, cls_val_loss = train_model(
            classifier, cls_train_loader, cls_val_loader, cls_loss, device,
            CLASS_LR, MAX_CLASS_EPOCHS, CLASS_PATIENCE, cls_path,
            f"{mode.title()} two-stage rainfall classifier", transfer,
        )

        # Conditional amount regressor
        set_seed(SEED + 20)
        regressor = GRUHead("regressor")
        if transfer:
            initialize_from_pretrained(regressor, source_checkpoint)
        reg_path = CHECKPOINT_DIR / f"two_stage_{mode}_amount.pt"
        regressor, histories[f"{mode}_amount"], reg_epoch, reg_val_loss = train_model(
            regressor, amount_train_loader, amount_val_loader, amount_loss, device,
            AMOUNT_LR, MAX_AMOUNT_EPOCHS, AMOUNT_PATIENCE, reg_path,
            f"{mode.title()} two-stage conditional amount regressor", transfer,
        )

        fitted[mode] = {
            "classifier": classifier,
            "regressor": regressor,
            "classifier_best_epoch": cls_epoch,
            "classifier_best_val_loss": cls_val_loss,
            "amount_best_epoch": reg_epoch,
            "amount_best_val_loss": reg_val_loss,
        }

    # Select probability thresholds using 2022 only; store the full threshold sweep.
    threshold_rows = []
    validation_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    selected_thresholds: dict[str, float] = {}

    for mode, models in fitted.items():
        val_logits = predict_raw(models["classifier"], X_val, device)
        val_prob = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -40, 40)))
        threshold, sweep = select_threshold(y_val, val_prob)
        sweep.insert(0, "model", mode)
        threshold_rows.append(sweep)
        selected_thresholds[mode] = threshold

        val_amount_log = predict_raw(models["regressor"], X_val, device)
        val_amount = np.maximum(np.expm1(np.maximum(val_amount_log, 0.0)), RAIN_THRESHOLD_MM)
        validation_predictions[mode] = gated_prediction(val_prob, val_amount, threshold)

        test_logits = predict_raw(models["classifier"], X_test, device)
        test_prob = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -40, 40)))
        test_amount_log = predict_raw(models["regressor"], X_test, device)
        test_amount = np.maximum(np.expm1(np.maximum(test_amount_log, 0.0)), RAIN_THRESHOLD_MM)
        test_predictions[mode] = gated_prediction(test_prob, test_amount, threshold)
        logging.info("%s: selected 2022 probability threshold=%.2f", mode, threshold)

    threshold_frame = pd.concat(threshold_rows, ignore_index=True)
    threshold_frame.to_csv(METRICS_DIR / "two_stage_threshold_sweep_2022.csv", index=False)

    val_rows = []
    for mode, pred in validation_predictions.items():
        val_rows.append({"model": f"two_stage_{mode}", "probability_threshold_selected_on_2022": selected_thresholds[mode],
                         **regression_metrics(y_val, pred)})
    pd.DataFrame(val_rows).to_csv(METRICS_DIR / "two_stage_validation_metrics_2022.csv", index=False)

    test_rows = []
    for mode, pred in test_predictions.items():
        test_rows.append({"model": f"two_stage_{mode}", "probability_threshold_selected_on_2022": selected_thresholds[mode],
                          **regression_metrics(y_test, pred)})
    two_stage_metrics = pd.DataFrame(test_rows)
    two_stage_metrics.to_csv(METRICS_DIR / "two_stage_metrics_test_2023.csv", index=False)

    # Append new models to the preserved original comparison, without modifying it.
    original_path = METRICS_DIR / "model_comparison_test_2023_with_zero_baseline.csv"
    if not original_path.exists():
        original_path = METRICS_DIR / "model_comparison_test_2023.csv"
    if original_path.exists():
        original = pd.read_csv(original_path)
        original_names = set(original["model"].astype(str))
        additions = two_stage_metrics.copy()
        additions = additions.rename(columns={"probability_threshold_selected_on_2022": "selected_probability_threshold_2022"})
        combined = pd.concat([original, additions], ignore_index=True, sort=False)
        combined.to_csv(METRICS_DIR / "model_comparison_all_models_test_2023.csv", index=False)
    else:
        logging.warning("Original comparison CSV not found; saving two-stage-only comparison.")
        two_stage_metrics.to_csv(METRICS_DIR / "model_comparison_all_models_test_2023.csv", index=False)

    pred_frame = pd.DataFrame({
        "date": arrays["dates_target_test"].astype(str),
        "latitude": arrays["latitude_target_test"].astype(float),
        "longitude": arrays["longitude_target_test"].astype(float),
        "imd_actual_mm": y_test,
    })
    for mode in ("scratch", "transfer"):
        pred_frame[f"two_stage_{mode}_probability"] = 1.0 / (1.0 + np.exp(-np.clip(
            predict_raw(fitted[mode]["classifier"], X_test, device), -40, 40)))
        pred_frame[f"two_stage_{mode}_threshold_2022"] = selected_thresholds[mode]
        pred_frame[f"two_stage_{mode}_prediction_mm"] = test_predictions[mode]
    pred_path = PREDICTIONS_DIR / "two_stage_test_2023_predictions.csv"
    pred_frame.to_csv(pred_path, index=False)

    # Full training histories and plots.
    save_histories(histories)
    plot_histories(histories)

    # Test predictions versus observations, full and low-range zoom.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    max_obs = float(np.max(y_test))
    for mode, label in (("scratch", "Two-stage scratch GRU"), ("transfer", "Two-stage transfer GRU")):
        axes[0].scatter(y_test, test_predictions[mode], s=7, alpha=0.22, label=label)
    axes[0].plot([0, max_obs], [0, max_obs], "k--", linewidth=1, label="Ideal")
    axes[0].set_xlim(0, max_obs)
    axes[0].set_ylim(0, max_obs)
    axes[0].set_xlabel("Observed IMD rainfall (mm/day)")
    axes[0].set_ylabel("Predicted rainfall (mm/day)")
    axes[0].set_title("2023 test: full scale (exploratory)")
    axes[0].legend(fontsize=8)

    zoom_limit = float(np.quantile(y_test, 0.99))
    for mode, label in (("scratch", "Two-stage scratch GRU"), ("transfer", "Two-stage transfer GRU")):
        mask = y_test <= zoom_limit
        axes[1].scatter(y_test[mask], test_predictions[mode][mask], s=7, alpha=0.22, label=label)
    axes[1].plot([0, zoom_limit], [0, zoom_limit], "k--", linewidth=1, label="Ideal")
    axes[1].set_xlim(0, zoom_limit)
    axes[1].set_ylim(0, zoom_limit)
    axes[1].set_xlabel("Observed IMD rainfall (mm/day)")
    axes[1].set_ylabel("Predicted rainfall (mm/day)")
    axes[1].set_title("2023 test: up to 99th percentile")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "two_stage_gru_test_scatter_2023.png", dpi=170)
    plt.close(fig)

    # Validation CSI threshold curves to show threshold choice transparently.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode in ("scratch", "transfer"):
        part = threshold_frame[threshold_frame["model"] == mode]
        ax.plot(part["threshold"], part["CSI"], label=f"{mode.title()} classifier")
        ax.axvline(selected_thresholds[mode], linestyle="--", alpha=0.55)
    ax.set_xlabel("Rainfall probability threshold")
    ax.set_ylabel("2022 validation CSI")
    ax.set_title("Classifier threshold selection using 2022 validation data")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "two_stage_threshold_selection_2022.png", dpi=170)
    plt.close(fig)

    payload = {
        "experiment": "Two-stage GRU rainfall occurrence and conditional amount prediction",
        "target_training_period": "2019-2021",
        "validation_period": "2022",
        "test_period": "2023 (previously inspected in the original experiment; report as exploratory/post-hoc)",
        "lookback_days": LOOKBACK,
        "input_features": INPUT_SIZE,
        "rainfall_event_threshold_mm": RAIN_THRESHOLD_MM,
        "classifier_loss": "weighted binary cross-entropy; positive class weight calculated from target_train only",
        "amount_loss": "SmoothL1/Huber on log1p rainfall, fitted only to target_train observations >= 2.5 mm/day",
        "prediction_rule": "Predict zero if event probability is below the model-specific threshold selected by maximum CSI on 2022 validation; otherwise predict conditional amount, clipped to at least 2.5 mm/day.",
        "threshold_selection": "Grid search from 0.05 to 0.95 in 0.01 increments; maximize 2022 validation CSI; tie-break by HSS, then lower FAR, then higher threshold.",
        "transfer_initialization": "Initialize GRU encoder weights from the previously trained ERA5-pretrained GRU; initialize a new output head for each stage.",
        "selected_thresholds_2022": selected_thresholds,
        "training": {mode: {
            "classifier_best_epoch": fitted[mode]["classifier_best_epoch"],
            "classifier_best_validation_loss": fitted[mode]["classifier_best_val_loss"],
            "amount_best_epoch": fitted[mode]["amount_best_epoch"],
            "amount_best_validation_loss": fitted[mode]["amount_best_val_loss"],
        } for mode in fitted},
        "test_metrics_2023": {row["model"]: json_safe(row) for row in test_rows},
        "caution": "The 2023 results are post-hoc/exploratory because 2023 was inspected during the previous model analysis. Do not claim a new untouched test evaluation.",
    }
    (METRICS_DIR / "two_stage_experiment_2023.json").write_text(
        json.dumps(json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")

    logging.info("TWO-STAGE TRAINING COMPLETED")
    logging.info("Classifier/regressor training events: %d / %d", n_pos, int(positive_train.sum()))
    for mode, pred in test_predictions.items():
        m = regression_metrics(y_test, pred)
        logging.info("two_stage_%s | MAE=%.4f | RMSE=%.4f | NSE=%s | KGE=%s | POD=%s | FAR=%s | CSI=%s",
                     mode, m["MAE_mm"], m["RMSE_mm"], f"{m['NSE']:.4f}", f"{m['KGE']:.4f}",
                     f"{m['POD']:.4f}", f"{m['FAR']:.4f}", f"{m['CSI']:.4f}")
    logging.info("Saved predictions: %s", pred_path)
    logging.info("Saved metrics: %s", METRICS_DIR / "two_stage_metrics_test_2023.csv")
    logging.info("Saved combined comparison: %s", METRICS_DIR / "model_comparison_all_models_test_2023.csv")
    logging.info("Saved figures: two_stage_gru_training_curves.png, two_stage_gru_test_scatter_2023.png, two_stage_threshold_selection_2022.png")


if __name__ == "__main__":
    main()
