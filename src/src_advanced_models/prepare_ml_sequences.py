#!/usr/bin/env python3
"""Prepare leakage-safe daily sequences for ERA5 pretraining and IMD transfer learning.

Task convention:
  Input: 7 consecutive days of ERA5 features ending on target_date - 1 day.
  Source label: ERA5 tp_mm on target_date.
  Target label: IMD rainfall on target_date.

Chronological partitions:
  ERA5 source train: 2014-2017
  ERA5 source validation: 2018
  IMD target train: 2019-2021
  IMD target validation: 2022
  IMD final test: 2023

The feature scaler is fitted only on ERA5 feature observations from 2014-2017.
No target-year labels are used to fit scaling parameters.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "processed" / "era5_imd_merged_2014_2023.nc"
OUTPUT_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "results" / "checkpoints"
METRICS_DIR = ROOT / "results" / "metrics"
LOG_PATH = ROOT / "logs" / "sequence_preparation.log"
OUTPUT_NPZ = OUTPUT_DIR / "ml_sequences_2014_2023.npz"
OUTPUT_JSON = METRICS_DIR / "ml_sequence_manifest_2014_2023.json"
SCALER_PATH = CHECKPOINT_DIR / "era5_feature_scaler_2014_2017.joblib"

LOOKBACK = 7
FEATURES = [
    "t2m_C", "d2m_C", "sp_hPa", "tp_mm", "u10_ms", "v10_ms",
    "tcc", "blh_m", "r700_pct", "u850_ms", "v850_ms",
    "wind_speed_10m_ms", "rh2m_pct", "doy_sin", "doy_cos",
]
SPLITS = {
    "source_train": (2014, 2017, "tp_mm"),
    "source_val": (2018, 2018, "tp_mm"),
    "target_train": (2019, 2021, "imd_rain_mm"),
    "target_val": (2022, 2022, "imd_rain_mm"),
    "target_test": (2023, 2023, "imd_rain_mm"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH, mode="w")],
        force=True,
    )


def main() -> None:
    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Merged input dataset not found: {INPUT_PATH}")

    logging.info("Loading merged ERA5 + IMD dataset: %s", INPUT_PATH)
    with xr.open_dataset(INPUT_PATH) as opened:
        required = set(FEATURES + ["imd_rain_mm"])
        missing = sorted(required - set(opened.data_vars))
        if missing:
            raise ValueError(f"Merged dataset is missing variables: {missing}")
        ds = opened[FEATURES + ["imd_rain_mm"]].load()

    for coord in ("time", "latitude", "longitude"):
        if coord not in ds.coords:
            raise ValueError(f"Required coordinate missing: {coord}")

    dates = pd.DatetimeIndex(pd.to_datetime(ds.time.values))
    expected_dates = pd.date_range("2014-01-01", "2023-12-31", freq="D")
    if not dates.equals(expected_dates):
        raise ValueError("Merged dataset dates are not a complete, ordered 2014-2023 daily index.")
    if ds.sizes.get("latitude") != 6 or ds.sizes.get("longitude") != 6:
        raise ValueError("Expected a 6x6 study grid.")

    latitudes = np.asarray(ds.latitude.values, dtype=np.float32)
    longitudes = np.asarray(ds.longitude.values, dtype=np.float32)
    feature_values = np.stack(
        [np.asarray(ds[name].transpose("time", "latitude", "longitude").values, dtype=np.float32)
         for name in FEATURES],
        axis=-1,
    )
    source_target = np.asarray(
        ds["tp_mm"].transpose("time", "latitude", "longitude").values,
        dtype=np.float32,
    )
    imd_target = np.asarray(
        ds["imd_rain_mm"].transpose("time", "latitude", "longitude").values,
        dtype=np.float32,
    )

    if not np.isfinite(feature_values).all():
        raise ValueError("ERA5 predictors contain NaN or infinite values.")
    if not np.isfinite(source_target).all() or not np.isfinite(imd_target).all():
        raise ValueError("One or more rainfall targets contain NaN or infinite values.")
    if (source_target < 0).any() or (imd_target < 0).any():
        raise ValueError("Rainfall targets must be non-negative.")

    # Fit scaling only on source-training-era feature values (2014-2017).
    source_fit_mask = (dates.year >= 2014) & (dates.year <= 2017)
    scaler = StandardScaler()
    scaler.fit(feature_values[source_fit_mask].reshape(-1, len(FEATURES)))
    joblib.dump(scaler, SCALER_PATH)
    logging.info("Feature scaler fitted only on 2014-2017 ERA5 predictors.")

    # Transform all features using the frozen source-period scaler.
    scaled = scaler.transform(feature_values.reshape(-1, len(FEATURES)))
    scaled = scaled.reshape(feature_values.shape).astype(np.float32)

    # Each sample belongs to one grid cell and one target date. X ends on D-1;
    # the label is rainfall observed/analysed on D. This is next-day prediction.
    collected: dict[str, dict[str, list[np.ndarray]]] = {
        name: {key: [] for key in ("X", "y_mm", "dates", "lat", "lon")}
        for name in SPLITS
    }
    for target_index in range(LOOKBACK, len(dates)):
        target_date = dates[target_index]
        year = target_date.year
        split_name = None
        for name, (first_year, last_year, _) in SPLITS.items():
            if first_year <= year <= last_year:
                split_name = name
                break
        if split_name is None:
            continue

        # Shape from (lookback, lat, lon, feature) to (36 cells, lookback, features).
        sequence = scaled[target_index - LOOKBACK:target_index]
        sequence = sequence.transpose(1, 2, 0, 3).reshape(
            len(latitudes) * len(longitudes), LOOKBACK, len(FEATURES)
        )
        target_name = SPLITS[split_name][2]
        label_grid = source_target[target_index] if target_name == "tp_mm" else imd_target[target_index]
        labels = label_grid.reshape(-1)

        collected[split_name]["X"].append(sequence)
        collected[split_name]["y_mm"].append(labels)
        collected[split_name]["dates"].append(
            np.repeat(np.datetime64(target_date.date(), "D"), labels.size)
        )
        collected[split_name]["lat"].append(np.repeat(latitudes, len(longitudes)))
        collected[split_name]["lon"].append(np.tile(longitudes, len(latitudes)))

    arrays: dict[str, np.ndarray] = {}
    report_splits = {}
    for name, contents in collected.items():
        if not contents["X"]:
            raise ValueError(f"No samples generated for split {name}.")
        X = np.concatenate(contents["X"], axis=0).astype(np.float32)
        y_mm = np.concatenate(contents["y_mm"], axis=0).astype(np.float32)
        sample_dates = np.concatenate(contents["dates"])
        sample_lat = np.concatenate(contents["lat"]).astype(np.float32)
        sample_lon = np.concatenate(contents["lon"]).astype(np.float32)
        if not np.isfinite(X).all() or not np.isfinite(y_mm).all():
            raise ValueError(f"Generated split {name} contains non-finite values.")
        if (y_mm < 0).any():
            raise ValueError(f"Generated split {name} contains negative rainfall.")

        arrays[f"X_{name}"] = X
        arrays[f"y_{name}_mm"] = y_mm
        arrays[f"y_{name}_log1p"] = np.log1p(y_mm).astype(np.float32)
        arrays[f"dates_{name}"] = sample_dates.astype("datetime64[D]").astype(str)
        arrays[f"latitude_{name}"] = sample_lat
        arrays[f"longitude_{name}"] = sample_lon

        report_splits[name] = {
            "target_variable": SPLITS[name][2],
            "first_target_date": str(pd.Timestamp(sample_dates.min()).date()),
            "last_target_date": str(pd.Timestamp(sample_dates.max()).date()),
            "samples": int(X.shape[0]),
            "sequence_shape_per_sample": [LOOKBACK, len(FEATURES)],
            "target_min_mm": float(y_mm.min()),
            "target_mean_mm": float(y_mm.mean()),
            "target_max_mm": float(y_mm.max()),
            "zero_target_percent": float(np.mean(y_mm == 0.0) * 100.0),
        }
        logging.info(
            "%s: %s to %s | %d samples | label=%s",
            name,
            report_splits[name]["first_target_date"],
            report_splits[name]["last_target_date"],
            X.shape[0],
            SPLITS[name][2],
        )

    expected_samples = {
        "source_train": 1454 * 36,
        "source_val": 365 * 36,
        "target_train": 1096 * 36,
        "target_val": 365 * 36,
        "target_test": 365 * 36,
    }
    for name, expected in expected_samples.items():
        actual = report_splits[name]["samples"]
        if actual != expected:
            raise ValueError(f"Unexpected sample count for {name}: {actual}; expected {expected}.")

    np.savez_compressed(OUTPUT_NPZ, **arrays)
    report = {
        "status": "PASS",
        "source_dataset": str(INPUT_PATH),
        "source_dataset_sha256": sha256(INPUT_PATH),
        "output_npz": str(OUTPUT_NPZ),
        "output_npz_sha256": sha256(OUTPUT_NPZ),
        "feature_scaler": str(SCALER_PATH),
        "feature_scaler_sha256": sha256(SCALER_PATH),
        "lookback_days": LOOKBACK,
        "input_features": FEATURES,
        "input_timing": "For target date D, inputs are ERA5 features from D-7 through D-1; no features from D are used.",
        "source_target": "ERA5 tp_mm on target date D, used only for source pretraining labels.",
        "transfer_target": "IMD imd_rain_mm on target date D, used for fine-tuning, validation and final testing.",
        "target_transform_for_training": "log1p(rainfall_mm); retain *_mm arrays for evaluation in physical units.",
        "scaler_fit_period": "2014-01-01 to 2017-12-31 only",
        "splits": report_splits,
        "total_samples": int(sum(item["samples"] for item in report_splits.values())),
        "warning": "2023 is the only final test year; conclusions about interannual generalization must be cautious.",
    }
    OUTPUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logging.info("SEQUENCE PREPARATION PASSED")
    logging.info("Total samples: %d", report["total_samples"])
    logging.info("Saved sequences: %s", OUTPUT_NPZ)
    logging.info("Saved scaler: %s", SCALER_PATH)
    logging.info("Saved manifest: %s", OUTPUT_JSON)
    logging.info("Saved log: %s", LOG_PATH)


if __name__ == "__main__":
    main()
