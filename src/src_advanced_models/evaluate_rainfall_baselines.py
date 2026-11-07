#!/usr/bin/env python3
"""Evaluate simple rainfall baselines and consolidate the exploratory 2023 results.

Baselines:
  1) Monthly grid-cell climatology: mean IMD rainfall for the same grid cell
     and calendar month, calculated from 2019-2021 target-training observations.
  2) Previous-day persistence: previous day's IMD rainfall at the same grid cell.
  3) Always-zero reference.

The 2023 period has already been inspected in earlier experiments. All 2023
metrics produced here are therefore labelled exploratory/post-hoc, not as a
new untouched test evaluation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "era5_imd_merged_2014_2023.nc"
PREDICTIONS_PATH = ROOT / "results" / "predictions" / "imd_test_2023_predictions.csv"
MASTER_PATH = ROOT / "results" / "metrics" / "master_model_comparison_2023.csv"

PRED_DIR = ROOT / "results" / "predictions"
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"
LOG_PATH = ROOT / "logs" / "baseline_evaluation.log"

TRAIN_START = pd.Timestamp("2019-01-01")
TRAIN_END = pd.Timestamp("2021-12-31")
TEST_START = pd.Timestamp("2023-01-01")
TEST_END = pd.Timestamp("2023-12-31")
RAIN_THRESHOLD_MM = 2.5
EXPECTED_TEST_ROWS = 365 * 36
EVALUATION_NOTE = "2023 exploratory/post-hoc; 2023 was inspected in earlier experiments"


def setup_logging() -> None:
    for folder in (PRED_DIR, METRICS_DIR, FIGURES_DIR, LOG_PATH.parent):
        folder.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH, mode="w")],
        force=True,
    )


def safe_divide(num: float, den: float) -> float:
    return float(num / den) if den != 0 else float("nan")


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray, model_name: str) -> dict:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)

    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch for {model_name}: {actual.shape} vs {predicted.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError(f"Non-finite values found for {model_name}")
    if (actual < 0).any() or (predicted < 0).any():
        raise ValueError(f"Negative rainfall found for {model_name}")

    error = predicted - actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    obs_mean = float(np.mean(actual))
    obs_variance = float(np.sum((actual - obs_mean) ** 2))
    nse = float(1.0 - np.sum(error ** 2) / obs_variance) if obs_variance > 0 else float("nan")

    obs_std = float(np.std(actual, ddof=0))
    pred_std = float(np.std(predicted, ddof=0))
    pred_mean = float(np.mean(predicted))
    # KGE is undefined if either series has zero variance or observed mean is zero.
    if obs_std > 0 and pred_std > 0 and obs_mean != 0:
        correlation = float(np.corrcoef(actual, predicted)[0, 1])
        alpha = pred_std / obs_std
        beta = pred_mean / obs_mean
        kge = float(1.0 - np.sqrt((correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    else:
        kge = float("nan")

    observed_event = actual >= RAIN_THRESHOLD_MM
    predicted_event = predicted >= RAIN_THRESHOLD_MM
    tp = int(np.sum(observed_event & predicted_event))
    fp = int(np.sum(~observed_event & predicted_event))
    fn = int(np.sum(observed_event & ~predicted_event))
    tn = int(np.sum(~observed_event & ~predicted_event))

    pod = safe_divide(tp, tp + fn)
    far = safe_divide(fp, tp + fp)
    csi = safe_divide(tp, tp + fp + fn)
    hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = safe_divide(2.0 * (tp * tn - fn * fp), hss_den)

    return {
        "model": model_name,
        "n_samples": int(len(actual)),
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
        "evaluation_status": EVALUATION_NOTE,
    }


def main() -> None:
    setup_logging()
    logging.info("Starting additional rainfall baseline evaluation.")

    for path in (DATA_PATH, PREDICTIONS_PATH, MASTER_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    # Load IMD observations from the merged NetCDF, retaining dates and grid cells.
    logging.info("Loading merged ERA5/IMD NetCDF: %s", DATA_PATH)
    with xr.open_dataset(DATA_PATH) as ds:
        required = {"time", "latitude", "longitude"}
        if not required.issubset(set(ds.coords)):
            raise ValueError(f"Expected coordinates {required}; found {list(ds.coords)}")
        if "imd_rain_mm" not in ds.data_vars:
            raise ValueError("Merged dataset does not contain imd_rain_mm")
        imd_all = ds["imd_rain_mm"].to_dataframe(name="imd_rain_mm").reset_index()

    imd_all["date"] = pd.to_datetime(imd_all["time"]).dt.normalize()
    imd_all = imd_all.drop(columns=["time"])
    key_cols = ["date", "latitude", "longitude"]
    if imd_all.duplicated(key_cols).any():
        raise ValueError("Duplicate date/grid-cell keys in IMD NetCDF values")
    if not np.isfinite(imd_all["imd_rain_mm"].to_numpy()).all():
        raise ValueError("IMD rainfall contains non-finite values")
    if (imd_all["imd_rain_mm"] < 0).any():
        raise ValueError("IMD rainfall contains negative values")

    # Read the actual 2023 predictions from the existing model run.
    logging.info("Loading existing 2023 predictions: %s", PREDICTIONS_PATH)
    existing = pd.read_csv(PREDICTIONS_PATH)
    required_prediction_cols = set(key_cols + ["imd_actual_mm"])
    if not required_prediction_cols.issubset(existing.columns):
        raise ValueError(
            f"Prediction CSV missing columns: {required_prediction_cols - set(existing.columns)}"
        )
    existing["date"] = pd.to_datetime(existing["date"]).dt.normalize()
    if existing.duplicated(key_cols).any():
        raise ValueError("Duplicate date/grid-cell keys in the original prediction CSV")

    target = existing.loc[
        (existing["date"] >= TEST_START) & (existing["date"] <= TEST_END)
    ].copy()
    if len(target) != EXPECTED_TEST_ROWS:
        raise ValueError(f"Expected {EXPECTED_TEST_ROWS} 2023 rows, found {len(target)}")

    # Confirm that CSV actual values match the canonical IMD NetCDF target.
    imd_2023 = imd_all.loc[
        (imd_all["date"] >= TEST_START) & (imd_all["date"] <= TEST_END),
        key_cols + ["imd_rain_mm"],
    ]
    target = target.merge(imd_2023, on=key_cols, how="left", validate="one_to_one")
    if target["imd_rain_mm"].isna().any():
        raise ValueError("Some 2023 prediction rows do not match an IMD NetCDF observation")
    if not np.allclose(target["imd_actual_mm"], target["imd_rain_mm"], rtol=1e-5, atol=1e-5):
        max_diff = float(np.max(np.abs(target["imd_actual_mm"] - target["imd_rain_mm"])))
        raise ValueError(f"CSV actual rainfall differs from NetCDF IMD values (max diff={max_diff})")
    target = target.drop(columns=["imd_rain_mm"])

    # Baseline A: monthly grid-cell climatology, trained only on 2019-2021 labels.
    train = imd_all.loc[
        (imd_all["date"] >= TRAIN_START) & (imd_all["date"] <= TRAIN_END)
    ].copy()
    if train.empty:
        raise ValueError("No 2019-2021 IMD training observations found")
    train["month"] = train["date"].dt.month
    target["month"] = target["date"].dt.month
    monthly_climatology = (
        train.groupby(["latitude", "longitude", "month"], as_index=False)["imd_rain_mm"]
        .mean()
        .rename(columns={"imd_rain_mm": "monthly_climatology_pred_mm"})
    )
    target = target.merge(
        monthly_climatology,
        on=["latitude", "longitude", "month"],
        how="left",
        validate="many_to_one",
    )
    if target["monthly_climatology_pred_mm"].isna().any():
        raise ValueError("Monthly climatology could not be assigned to every 2023 grid-cell day")

    # Baseline B: yesterday's observed IMD rainfall predicts today's rainfall.
    # Shifting the source observation date forward by one day avoids using the
    # target day's observed rainfall as an input to its own prediction.
    persistence_source = imd_all[["date", "latitude", "longitude", "imd_rain_mm"]].copy()
    persistence_source["date"] = persistence_source["date"] + pd.Timedelta(days=1)
    persistence_source = persistence_source.rename(columns={"imd_rain_mm": "persistence_pred_mm"})
    target = target.merge(
        persistence_source.loc[
            (persistence_source["date"] >= TEST_START)
            & (persistence_source["date"] <= TEST_END)
        ],
        on=key_cols,
        how="left",
        validate="one_to_one",
    )
    if target["persistence_pred_mm"].isna().any():
        raise ValueError("Previous-day persistence baseline is missing observations")

    target["always_zero_pred_mm"] = 0.0
    target = target.drop(columns=["month"])
    target = target.sort_values(key_cols).reset_index(drop=True)

    prediction_output = PRED_DIR / "additional_baseline_predictions_2023.csv"
    target.to_csv(prediction_output, index=False, float_format="%.8f")

    # Calculate baselines on the exact same 2023 rows and targets as the GRUs.
    actual = target["imd_actual_mm"].to_numpy(dtype=np.float64)
    baseline_specs = [
        ("monthly_grid_cell_climatology", "monthly_climatology_pred_mm"),
        ("previous_day_persistence", "persistence_pred_mm"),
        ("always_zero_rainfall", "always_zero_pred_mm"),
    ]
    rows = []
    for model_name, prediction_col in baseline_specs:
        row = calculate_metrics(actual, target[prediction_col].to_numpy(), model_name)
        rows.append(row)
        logging.info(
            "%s | MAE=%.4f | RMSE=%.4f | NSE=%.4f | CSI=%.4f",
            model_name, row["MAE_mm"], row["RMSE_mm"], row["NSE"], row["CSI"],
        )

    baseline_metrics = pd.DataFrame(rows)
    baseline_metrics.to_csv(METRICS_DIR / "additional_baseline_comparison_2023.csv", index=False)
    baseline_metrics.loc[baseline_metrics["model"] == "monthly_grid_cell_climatology"].to_csv(
        METRICS_DIR / "monthly_climatology_metrics_2023.csv", index=False
    )
    baseline_metrics.loc[baseline_metrics["model"] == "previous_day_persistence"].to_csv(
        METRICS_DIR / "persistence_metrics_2023.csv", index=False
    )

    # Consolidate the 11 existing model rows with these three baselines.
    master = pd.read_csv(MASTER_PATH)
    required_master_cols = ["model", "n_samples", "MAE_mm", "RMSE_mm", "NSE", "KGE", "POD", "FAR", "CSI", "HSS"]
    missing_master = set(required_master_cols) - set(master.columns)
    if missing_master:
        raise ValueError(f"Master comparison missing expected columns: {missing_master}")

    new_names = [name for name, _ in baseline_specs]
    master = master.loc[~master["model"].isin(new_names)].copy()
    # Normalize evaluation-status naming across sources.
    if "evaluation_status" not in master.columns:
        master["evaluation_status"] = EVALUATION_NOTE
    else:
        master["evaluation_status"] = EVALUATION_NOTE

    combined = pd.concat([master, baseline_metrics], ignore_index=True, sort=False)
    combined["evaluation_status"] = EVALUATION_NOTE
    combined = combined.drop_duplicates(subset=["model"], keep="last")
    combined = combined.sort_values("CSI", ascending=False, na_position="last").reset_index(drop=True)

    # Make undefined KGE explicit for the always-zero forecast.
    combined.loc[combined["model"] == "always_zero_rainfall", "KGE"] = np.nan
    final_master_path = METRICS_DIR / "final_master_model_comparison_2023.csv"
    combined.to_csv(final_master_path, index=False, float_format="%.8f")

    # Comparison plot: amount errors and event detection skill on separate axes.
    plot_df = combined.copy()
    plot_df = plot_df.sort_values("CSI", ascending=True, na_position="first")
    labels = plot_df["model"].astype(str).tolist()
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(16, max(7, 0.55 * len(labels))))
    axes[0].barh(y - 0.18, plot_df["MAE_mm"], height=0.35, label="MAE")
    axes[0].barh(y + 0.18, plot_df["RMSE_mm"], height=0.35, label="RMSE")
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel("Error (mm/day)")
    axes[0].set_title("Rainfall amount error")
    axes[0].legend()
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(y - 0.18, plot_df["POD"], height=0.35, label="POD")
    axes[1].barh(y + 0.18, plot_df["CSI"], height=0.35, label="CSI")
    axes[1].set_yticks(y, labels)
    axes[1].set_xlabel("Score")
    axes[1].set_title(f"Rainfall-event detection (threshold = {RAIN_THRESHOLD_MM} mm/day)")
    axes[1].set_xlim(left=0)
    axes[1].legend()
    axes[1].grid(axis="x", alpha=0.25)
    fig.suptitle("Model and baseline comparison — 2023 exploratory evaluation", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "baseline_model_comparison_2023.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Reproducibility metadata.
    report = {
        "period_evaluated": "2023-01-01 to 2023-12-31",
        "evaluation_status": EVALUATION_NOTE,
        "n_grid_cell_days": int(len(target)),
        "event_threshold_mm_per_day": RAIN_THRESHOLD_MM,
        "monthly_climatology_training_period": "2019-01-01 to 2021-12-31",
        "monthly_climatology_method": "Mean IMD rainfall by grid cell and calendar month, using target-training period only.",
        "persistence_method": "IMD rainfall at the same grid cell on date D-1 is used to predict date D.",
        "persistence_caveat": "A reference benchmark; operational comparability depends on whether the previous-day IMD observation is available at forecast issuance time.",
        "all_zero_method": "Predict 0 mm/day for every grid-cell day.",
        "metrics": rows,
        "final_master_model_comparison": str(final_master_path),
        "prediction_file": str(prediction_output),
    }
    with (METRICS_DIR / "additional_baseline_evaluation_2023.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=True)

    logging.info("BASELINE EVALUATION COMPLETED")
    logging.info("2023 grid-cell-day rows: %d", len(target))
    logging.info("Monthly climatology training labels: 2019-2021 only")
    logging.info("Saved baseline predictions: %s", prediction_output)
    logging.info("Saved baseline metrics: %s", METRICS_DIR / "additional_baseline_comparison_2023.csv")
    logging.info("Saved final master table: %s", final_master_path)
    logging.info("Saved comparison figure: %s", FIGURES_DIR / "baseline_model_comparison_2023.png")

    print("\nADDITIONAL BASELINE RESULTS")
    print(baseline_metrics[["model", "MAE_mm", "RMSE_mm", "NSE", "KGE", "POD", "FAR", "CSI", "HSS"]].round(4).to_string(index=False))
    print("\nFINAL MASTER COMPARISON")
    print(combined[["model", "MAE_mm", "RMSE_mm", "NSE", "KGE", "POD", "FAR", "CSI", "HSS"]].round(4).to_string(index=False))
    print(f"\nTotal models/baselines in final table: {len(combined)}")


if __name__ == "__main__":
    main()
