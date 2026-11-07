#!/usr/bin/env python3
"""Align validated ERA5 daily features with IMD daily rainfall for 2014-2023.

This script does not resample, interpolate, fill, or otherwise alter rainfall.
It requires exact date alignment and matching 6x6 grid coordinates.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
ERA5_DIR = ROOT / "data" / "interim" / "era5_daily"
IMD_PATH = ROOT / "data" / "raw" / "imd" / "imd_rain_2000_2023.nc"
OUTPUT_DIR = ROOT / "data" / "processed"
METRICS_DIR = ROOT / "results" / "metrics"
OUTPUT_PATH = OUTPUT_DIR / "era5_imd_merged_2014_2023.nc"
REPORT_PATH = METRICS_DIR / "era5_imd_alignment_2014_2023.json"
LOG_PATH = ROOT / "logs" / "era5_imd_alignment.log"

START_YEAR = 2014
END_YEAR = 2023
ERA5_VARIABLES = [
    "t2m_C", "d2m_C", "sp_hPa", "tp_mm", "u10_ms", "v10_ms",
    "tcc", "blh_m", "r700_pct", "u850_ms", "v850_ms",
    "wind_speed_10m_ms", "rh2m_pct", "doy_sin", "doy_cos",
]
EXPECTED_LAT = np.array([13.75, 14.00, 14.25, 14.50, 14.75, 15.00])
EXPECTED_LON = np.array([76.75, 77.00, 77.25, 77.50, 77.75, 78.00])


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH, mode="w")],
    )


def find_coord_name(ds: xr.Dataset, choices: tuple[str, ...], label: str) -> str:
    for name in choices:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError(
        f"Could not identify {label} coordinate. Available coordinates: "
        f"{list(ds.coords)}; dimensions: {list(ds.dims)}"
    )


def normalize_spatial_names(ds: xr.Dataset) -> xr.Dataset:
    lat_name = find_coord_name(ds, ("latitude", "lat"), "latitude")
    lon_name = find_coord_name(ds, ("longitude", "lon"), "longitude")
    rename = {}
    if lat_name != "latitude":
        rename[lat_name] = "latitude"
    if lon_name != "longitude":
        rename[lon_name] = "longitude"
    if rename:
        ds = ds.rename(rename)
    return ds


def safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(values.size),
        "zero_percent": float(np.mean(values == 0.0) * 100.0),
        "positive_percent": float(np.mean(values > 0.0) * 100.0),
        "mean_mm": float(np.mean(values)),
        "std_mm": float(np.std(values)),
        "p95_mm": float(np.percentile(values, 95)),
        "p99_mm": float(np.percentile(values, 99)),
        "maximum_mm": float(np.max(values)),
        "percent_at_or_above_thresholds": {
            str(threshold): float(np.mean(values >= threshold) * 100.0)
            for threshold in (0.1, 1.0, 2.5, 15.6, 64.5)
        },
    }


def main() -> None:
    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    expected_dates = pd.date_range(
        f"{START_YEAR}-01-01", f"{END_YEAR}-12-31", freq="D"
    )
    expected_time_count = len(expected_dates)
    expected_years = list(range(START_YEAR, END_YEAR + 1))

    logging.info("Preparing ERA5 + IMD dataset for %s-%s", START_YEAR, END_YEAR)
    logging.info("Reading only the requested 10 annual ERA5 files; older files are ignored.")

    era5_parts: list[xr.Dataset] = []
    for year in expected_years:
        path = ERA5_DIR / f"era5_daily_{year}.nc"
        if not path.exists():
            raise FileNotFoundError(f"Missing ERA5 annual file: {path}")
        logging.info("Opening %s", path.name)
        with xr.open_dataset(path) as ds:
            missing = sorted(set(ERA5_VARIABLES) - set(ds.data_vars))
            if missing:
                raise ValueError(f"{path.name} is missing variables: {missing}")
            part = ds[ERA5_VARIABLES].load()
        era5_parts.append(part)

    era5 = xr.concat(era5_parts, dim="time", join="exact")
    era5 = normalize_spatial_names(era5)
    era5 = era5.sortby("time")

    era5_dates = pd.DatetimeIndex(pd.to_datetime(era5.time.values))
    if not era5_dates.equals(expected_dates):
        raise ValueError(
            "ERA5 dates do not exactly match the expected daily 2014-2023 index."
        )
    if era5.sizes.get("latitude") != 6 or era5.sizes.get("longitude") != 6:
        raise ValueError("ERA5 does not contain the expected 6x6 spatial grid.")
    if not np.allclose(era5.latitude.values, EXPECTED_LAT, atol=1e-6):
        raise ValueError("ERA5 latitude coordinates differ from the expected grid.")
    if not np.allclose(era5.longitude.values, EXPECTED_LON, atol=1e-6):
        raise ValueError("ERA5 longitude coordinates differ from the expected grid.")

    for variable in ERA5_VARIABLES:
        vals = np.asarray(era5[variable].values)
        if not np.isfinite(vals).all():
            raise ValueError(f"ERA5 variable {variable} contains NaN or infinite values.")
    if float(era5["tp_mm"].min()) < 0:
        raise ValueError("ERA5 tp_mm contains negative rainfall.")

    if not IMD_PATH.exists():
        raise FileNotFoundError(f"IMD file not found: {IMD_PATH}")
    logging.info("Opening IMD rainfall file %s", IMD_PATH.name)
    with xr.open_dataset(IMD_PATH) as raw_imd:
        if "rain" not in raw_imd.data_vars:
            raise ValueError(
                f"Expected IMD variable 'rain'; found {list(raw_imd.data_vars)}"
            )
        imd = normalize_spatial_names(raw_imd[["rain"]]).load()

    if "time" not in imd.coords and "time" not in imd.dims:
        raise ValueError("IMD dataset has no time coordinate/dimension.")
    imd = imd.sel(time=slice(expected_dates[0], expected_dates[-1]))
    imd_dates = pd.DatetimeIndex(pd.to_datetime(imd.time.values))
    if not imd_dates.equals(expected_dates):
        raise ValueError(
            "IMD dates do not exactly match ERA5 dates for 2014-2023. "
            f"Found {len(imd_dates)} dates; expected {expected_time_count}."
        )
    if imd.sizes.get("latitude") != 6 or imd.sizes.get("longitude") != 6:
        raise ValueError("IMD does not contain the expected 6x6 spatial grid.")
    if not np.allclose(imd.latitude.values, EXPECTED_LAT, atol=1e-6):
        raise ValueError("IMD latitude coordinates differ from the ERA5 grid.")
    if not np.allclose(imd.longitude.values, EXPECTED_LON, atol=1e-6):
        raise ValueError("IMD longitude coordinates differ from the ERA5 grid.")

    imd_rain = imd["rain"].transpose("time", "latitude", "longitude")
    imd_values = np.asarray(imd_rain.values, dtype=np.float64)
    if not np.isfinite(imd_values).all():
        raise ValueError("IMD rainfall contains NaN or infinite values in 2014-2023.")
    if float(np.min(imd_values)) < 0:
        raise ValueError("IMD rainfall contains negative values.")

    # Rebind identical, already-verified coordinates to avoid accidental nearest-cell
    # matching or implicit alignment by coordinate labels.
    merged = era5.assign(
        imd_rain_mm=(
            ("time", "latitude", "longitude"),
            imd_values,
        )
    )
    merged.attrs.update({
        "description": "Daily ERA5 atmospheric predictors aligned with IMD gridded rainfall observations.",
        "period": "2014-01-01 to 2023-12-31",
        "study_grid": "6x6 grid; latitude 13.75-15.00; longitude 76.75-78.00; 0.25 degree spacing",
        "daily_alignment": "ERA5 daily aggregation uses the IMD observation-day window documented in the extraction script.",
        "target_variable": "imd_rain_mm",
        "prediction_task": "Use historical ERA5 predictor sequences to predict next-day IMD rainfall; sequence shifting is performed in the ML dataset preparation stage.",
        "note": "ERA5 tp_mm is a source-product precipitation feature/diagnostic, not the IMD target label.",
    })

    for variable in list(merged.data_vars):
        if merged[variable].dtype.kind == "f":
            merged[variable] = merged[variable].astype("float32")

    encoding = {
        variable: {"zlib": True, "complevel": 4}
        for variable in merged.data_vars
    }
    logging.info("Writing merged dataset to %s", OUTPUT_PATH)
    merged.to_netcdf(OUTPUT_PATH, engine="netcdf4", encoding=encoding)

    era5_rain = np.asarray(merged["tp_mm"].values, dtype=np.float64).reshape(-1)
    imd_flat = imd_values.reshape(-1)
    thresholds = {}
    for threshold in (0.1, 1.0, 2.5, 15.6, 64.5):
        era_evt = era5_rain >= threshold
        imd_evt = imd_flat >= threshold
        tp = int(np.sum(era_evt & imd_evt))
        fp = int(np.sum(era_evt & ~imd_evt))
        fn = int(np.sum(~era_evt & imd_evt))
        tn = int(np.sum(~era_evt & ~imd_evt))
        thresholds[str(threshold)] = {
            "era5_event_percent": float(np.mean(era_evt) * 100),
            "imd_event_percent": float(np.mean(imd_evt) * 100),
            "both_event_count": tp,
            "era5_only_event_count": fp,
            "imd_only_event_count": fn,
            "neither_event_count": tn,
            "event_overlap_csi_descriptive_only": (
                float(tp / (tp + fp + fn)) if (tp + fp + fn) else None
            ),
        }

    report = {
        "status": "PASS",
        "period": "2014-01-01 to 2023-12-31",
        "daily_dates": int(expected_time_count),
        "grid_cells": 36,
        "grid_cell_day_records": int(expected_time_count * 36),
        "predictor_variables": ERA5_VARIABLES,
        "target_variable": "imd_rain_mm",
        "merged_file": str(OUTPUT_PATH),
        "merged_file_size_bytes": int(OUTPUT_PATH.stat().st_size),
        "era5_rainfall_distribution_mm": distribution(era5_rain),
        "imd_rainfall_distribution_mm": distribution(imd_flat),
        "era5_imd_paired_rainfall_correlation": safe_corr(era5_rain, imd_flat),
        "same_threshold_product_comparison_not_model_skill": thresholds,
        "interpretation_warning": (
            "ERA5 and IMD are different precipitation products. Product agreement metrics here are descriptive only, "
            "not model forecast scores. Exact-zero frequencies should not be treated as directly equivalent; "
            "compare distributions and event frequencies at common rainfall thresholds."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logging.info("MERGE AND ALIGNMENT PASSED")
    logging.info("Dates: %s to %s (%d)", expected_dates[0].date(), expected_dates[-1].date(), expected_time_count)
    logging.info("Grid cells: 36; grid-cell-day records: %d", expected_time_count * 36)
    logging.info("Predictors: %d; IMD target: imd_rain_mm", len(ERA5_VARIABLES))
    logging.info("ERA5 exact-zero rainfall: %.2f%%", report["era5_rainfall_distribution_mm"]["zero_percent"])
    logging.info("IMD exact-zero rainfall: %.2f%%", report["imd_rainfall_distribution_mm"]["zero_percent"])
    logging.info("Paired rainfall correlation (descriptive): %s", report["era5_imd_paired_rainfall_correlation"])
    logging.info("Saved merged NetCDF: %s", OUTPUT_PATH)
    logging.info("Saved alignment report: %s", REPORT_PATH)
    logging.info("Saved log: %s", LOG_PATH)


if __name__ == "__main__":
    main()
