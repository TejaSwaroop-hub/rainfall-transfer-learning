#!/usr/bin/env python3
"""Run a reproducible checkpoint-based demo of next-day IMD rainfall prediction.

Place this file in the project's src/ directory and run:
    python src/demo_next_day_prediction.py

The script performs inference with the three already-trained original-GRU
checkpoints: ERA5 zero-shot, IMD scratch, and ERA5-to-IMD transfer. It does not
retrain models or modify checkpoints.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_transfer_models import GRURegressor, calculate_metrics, log_to_mm  # noqa: E402

NPZ_PATH = ROOT / "data" / "processed" / "ml_sequences_2014_2023.npz"
CHECKPOINTS = {
    "zero_shot_era5_gru": ROOT / "results" / "checkpoints" / "era5_pretrained_gru.pt",
    "scratch_gru": ROOT / "results" / "checkpoints" / "scratch_gru_imd.pt",
    "transfer_gru": ROOT / "results" / "checkpoints" / "transfer_gru_imd.pt",
}
DISPLAY_NAMES = {
    "zero_shot_era5_gru": "Zero-shot ERA5 GRU",
    "scratch_gru": "Scratch IMD GRU",
    "transfer_gru": "Transfer GRU",
}
OUTPUT_DIR = ROOT / "results" / "predictions"
BATCH_SIZE = 512
RAIN_THRESHOLD_MM = 2.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demonstrate next-day IMD rainfall prediction.")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="Inference device (default: auto).",
    )
    parser.add_argument(
        "--examples", type=int, default=5,
        help="Maximum number of representative examples to print (default: 5).",
    )
    return parser.parse_args()


def choose_example(mask: np.ndarray, rainfall: np.ndarray, target: float | None = None) -> int | None:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    if target is None:
        return int(indices[0])
    return int(indices[np.argmin(np.abs(rainfall[indices] - target))])


def load_checkpoint(path: Path, device: torch.device) -> tuple[GRURegressor, dict]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older PyTorch versions.
        checkpoint = torch.load(path, map_location="cpu")
    model = GRURegressor(
        input_size=int(checkpoint.get("input_size", 15)),
        hidden_size=int(checkpoint.get("hidden_size", 64)),
    ).to(device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def predict(model: GRURegressor, X_test: np.ndarray, device: torch.device) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.ascontiguousarray(X_test, dtype=np.float32))),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            outputs.append(model(batch_x).detach().cpu().numpy())
    return log_to_mm(np.concatenate(outputs).astype(np.float64))


def main() -> None:
    args = parse_args()
    print("=" * 78)
    print("NEXT-DAY RAINFALL PREDICTION - LIVE CHECKPOINT DEMONSTRATION")
    print("=" * 78)
    print("Task: use the previous 7 days of ERA5 features to estimate next-day IMD rainfall.")
    print(f"Prepared sequences: {NPZ_PATH}")

    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"Prepared sequence file not found: {NPZ_PATH}")
    missing_checkpoints = [str(path) for path in CHECKPOINTS.values() if not path.exists()]
    if missing_checkpoints:
        raise FileNotFoundError(
            "The following trained checkpoints are missing:\n  " + "\n  ".join(missing_checkpoints)
            + "\nRun the original training script first: python src/train_transfer_models.py"
        )

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device.")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"PyTorch: {torch.__version__}")
    print(f"Inference device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    with np.load(NPZ_PATH, allow_pickle=False) as data:
        required = (
            "X_target_test", "y_target_test_mm", "dates_target_test",
            "latitude_target_test", "longitude_target_test",
        )
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"Sequence file is missing arrays: {missing}")
        X_test = np.asarray(data["X_target_test"], dtype=np.float32)
        y_test = np.asarray(data["y_target_test_mm"], dtype=np.float64)
        dates = data["dates_target_test"].astype(str)
        latitudes = np.asarray(data["latitude_target_test"], dtype=np.float64)
        longitudes = np.asarray(data["longitude_target_test"], dtype=np.float64)

    if X_test.ndim != 3 or X_test.shape[1:] != (7, 15):
        raise ValueError(f"Expected test input shape (N, 7, 15), received {X_test.shape}.")
    if not np.isfinite(X_test).all() or not np.isfinite(y_test).all():
        raise ValueError("Test data contains NaN or infinite values.")
    if not (len(X_test) == len(y_test) == len(dates) == len(latitudes) == len(longitudes)):
        raise ValueError("Input, target, date, and coordinate arrays have different lengths.")

    print("\n[1/3] Prepared test sequences loaded")
    print(f"  Samples: {len(X_test):,}")
    print(f"  Sequence shape per sample: {X_test.shape[1]} days x {X_test.shape[2]} features")
    # NumPy 2.x does not reliably implement min/max reductions for Unicode arrays.
    # Parse the ISO-like date strings with pandas before reporting the range.
    parsed_dates = pd.to_datetime(dates, errors="raise")
    print(f"  Evaluation dates: {parsed_dates.min().date()} to {parsed_dates.max().date()}")
    print(f"  IMD zero-rainfall fraction: {np.mean(y_test == 0.0) * 100:.2f}%")

    predictions: dict[str, np.ndarray] = {}
    print("\n[2/3] Loading the three trained original-GRU checkpoints")
    for key, checkpoint_path in CHECKPOINTS.items():
        model, metadata = load_checkpoint(checkpoint_path, device)
        predictions[key] = predict(model, X_test, device)
        details = f"{DISPLAY_NAMES[key]}: loaded"
        if "best_epoch" in metadata:
            details += f", best validation epoch={metadata['best_epoch']}"
        if "best_validation_mse_log1p" in metadata:
            details += f", val MSE(log1p)={metadata['best_validation_mse_log1p']:.6f}"
        print(f"  {details}")
        del model

    print("\n[3/3] 2023 retrospective inference completed")
    print("  NOTE: 2023 is exploratory/post-hoc, not an untouched blind test set.")
    metrics_rows = []
    for key, values in predictions.items():
        metrics = calculate_metrics(y_test, values)
        metrics_rows.append({"model": DISPLAY_NAMES[key], **metrics})
    metrics_frame = pd.DataFrame(metrics_rows)
    metric_cols = ["model", "MAE_mm", "RMSE_mm", "NSE", "KGE", "POD", "FAR", "CSI", "HSS"]
    print("\nMetrics computed live from checkpoint predictions:")
    print(metrics_frame[metric_cols].to_string(index=False, formatters={
        c: (lambda x: f"{x:.4f}" if np.isfinite(x) else "undefined")
        for c in metric_cols if c != "model"
    }))
    print(f"\nCommon event threshold: {RAIN_THRESHOLD_MM:.1f} mm/day")
    print(f"Maximum observed rainfall: {y_test.max():.2f} mm/day")
    print(f"Maximum predicted rainfall (transfer GRU): {predictions['transfer_gru'].max():.2f} mm/day")

    groups = [
        ("Dry day", y_test == 0.0, None),
        ("Trace/light rain", (y_test > 0.0) & (y_test < 2.5), 1.0),
        ("Rain event", (y_test >= 2.5) & (y_test < 15.6), 5.0),
        ("Heavy rain", y_test >= 15.6, 25.0),
    ]
    chosen: list[tuple[str, int]] = []
    used: set[int] = set()
    for label, mask, target in groups:
        index = choose_example(mask, y_test, target)
        if index is not None and index not in used:
            chosen.append((label, index))
            used.add(index)
    wettest = int(np.argmax(y_test))
    if wettest not in used:
        chosen.append(("Wettest observation", wettest))
    else:
        for position, (label, index) in enumerate(chosen):
            if index == wettest:
                chosen[position] = ("Wettest observation", index)
                break
    chosen = chosen[:max(1, args.examples)]

    sample_rows = []
    for label, index in chosen:
        sample_rows.append({
            "example": label,
            "date": dates[index],
            "latitude": latitudes[index],
            "longitude": longitudes[index],
            "IMD_actual_mm": y_test[index],
            "zero_shot_ERA5_mm": predictions["zero_shot_era5_gru"][index],
            "scratch_GRU_mm": predictions["scratch_gru"][index],
            "transfer_GRU_mm": predictions["transfer_gru"][index],
        })
    sample_frame = pd.DataFrame(sample_rows)
    print("\nRepresentative grid-cell/day predictions (all values in mm/day):")
    print(sample_frame.to_string(index=False, formatters={
        "latitude": lambda x: f"{x:.2f}",
        "longitude": lambda x: f"{x:.2f}",
        "IMD_actual_mm": lambda x: f"{x:.2f}",
        "zero_shot_ERA5_mm": lambda x: f"{x:.2f}",
        "scratch_GRU_mm": lambda x: f"{x:.2f}",
        "transfer_GRU_mm": lambda x: f"{x:.2f}",
    }))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "demo_original_gru_comparison_2023.csv"
    full_frame = pd.DataFrame({
        "date": dates,
        "latitude": latitudes,
        "longitude": longitudes,
        "IMD_actual_mm": y_test,
        "zero_shot_ERA5_GRU_mm": predictions["zero_shot_era5_gru"],
        "scratch_IMD_GRU_mm": predictions["scratch_gru"],
        "transfer_GRU_mm": predictions["transfer_gru"],
        "actual_event_2_5mm": (y_test >= RAIN_THRESHOLD_MM).astype(int),
        "transfer_predicted_event_2_5mm": (predictions["transfer_gru"] >= RAIN_THRESHOLD_MM).astype(int),
    })
    full_frame.to_csv(output_path, index=False)
    print(f"\nSaved full 2023 demo predictions: {output_path}")
    print("Demo complete. No model was retrained and no checkpoint was modified.")


if __name__ == "__main__":
    main()
