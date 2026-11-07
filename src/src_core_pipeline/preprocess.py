"""Data Preprocessing Module for Rainfall Transfer Learning.
Calculates derived features (RH2m via Magnus formula, wind speed, cyclical day of year),
computes normalization statistics strictly on the ERA5 pre-training training split,
transforms targets y' = ln(1 + y), and saves processed partitions.
"""

import os
import argparse
import json
import yaml
import numpy as np
import pandas as pd
import xarray as xr

def magnus_rh(t2m_c: np.ndarray, d2m_c: np.ndarray) -> np.ndarray:
    """Computes relative humidity (%) from temperature and dewpoint in Celsius using Magnus formula."""
    a = 17.67
    b = 243.5
    es = 6.112 * np.exp((a * t2m_c) / (t2m_c + b))
    e = 6.112 * np.exp((a * d2m_c) / (d2m_c + b))
    rh = 100.0 * (e / es)
    return np.clip(rh, 0.0, 100.0)

def compute_class_distribution(rain_values: np.ndarray, thresholds: dict) -> dict:
    """Computes real distribution shares of rainfall classes on a pool of daily values."""
    rain = np.asarray(rain_values).flatten()
    # Remove NaNs
    rain = rain[~np.isnan(rain)]
    total = len(rain)
    
    shares = {}
    counts = {}
    for c_name, bounds in thresholds.items():
        low, high = float(bounds[0]), float(bounds[1])
        if high >= 1e8:
            mask = rain >= low
        elif low == 0.0:
            mask = rain < high
        else:
            mask = (rain >= low) & (rain < high)
        cnt = int(np.sum(mask))
        counts[c_name] = cnt
        shares[c_name] = float(cnt / total)
        
    return {"counts": counts, "shares": shares, "total_samples": total}

def preprocess_all(config_path: str = "configs/default.yaml", fast: bool = False):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    processed_dir = cfg["paths"]["processed"]
    os.makedirs(processed_dir, exist_ok=True)
    
    # Load raw NetCDF datasets
    era5_nc = os.path.join(cfg["paths"]["raw_era5"], "era5_anantapur_1979_2023.nc")
    imd_nc = os.path.join(cfg["paths"]["raw_imd"], "imd_rain_2000_2023.nc")
    
    print(f"Loading raw datasets:\n  ERA5: {era5_nc}\n  IMD:  {imd_nc}")
    ds_era5 = xr.open_dataset(era5_nc)
    ds_imd = xr.open_dataset(imd_nc)

    dates_era5 = pd.to_datetime(ds_era5.time.values)
    dates_imd = pd.to_datetime(ds_imd.time.values)
    lats = ds_imd.lat.values
    lons = ds_imd.lon.values
    n_cells = len(lats) * len(lons)
    
    print(f"Study Domain: {len(lats)} lats x {len(lons)} lons = {n_cells} cells.")

    # 1. Compute 4 Derived Variables for ERA5
    print("Computing derived meteorological variables (RH2m, wind speed, cyclical DOY)...")
    t2m = ds_era5.t2m.values
    d2m = ds_era5.d2m.values
    u10 = ds_era5.u10.values
    v10 = ds_era5.v10.values
    
    rh2m = magnus_rh(t2m, d2m)
    wind_speed = np.sqrt(u10**2 + v10**2)
    
    doy = dates_era5.dayofyear.values
    sin_doy_1d = np.sin(2.0 * np.pi * doy / 365.25)
    cos_doy_1d = np.cos(2.0 * np.pi * doy / 365.25)
    # Broadcast to (time, lat, lon)
    sin_doy = np.broadcast_to(sin_doy_1d[:, None, None], t2m.shape)
    cos_doy = np.broadcast_to(cos_doy_1d[:, None, None], t2m.shape)

    # Compile the 15 candidate predictors in exact configured order
    all_15_names = cfg["predictors"]["all_15"]
    predictor_arrays = {
        "tp": np.maximum(ds_era5.tp.values, 0.0),
        "tcwv": ds_era5.tcwv.values,
        "d2m": d2m,
        "t2m": t2m,
        "sp": ds_era5.sp.values,
        "cape": ds_era5.cape.values,
        "u10": u10,
        "v10": v10,
        "rh700": ds_era5.rh700.values,
        "u850": ds_era5.u850.values,
        "v850": ds_era5.v850.values,
        "RH2m": rh2m,
        "wind_speed": wind_speed,
        "sin_doy": sin_doy,
        "cos_doy": cos_doy,
    }

    # Stack into 4D array: (time, lat, lon, 15)
    era5_stacked = np.stack([predictor_arrays[name] for name in all_15_names], axis=-1)
    
    # 2. Strict Partition Masking
    pt_train_cfg = cfg["partitions"]["era5_pretrain_train"]
    pt_val_cfg = cfg["partitions"]["era5_pretrain_val"]
    
    mask_pt_train = (dates_era5.year >= pt_train_cfg["start_year"]) & (dates_era5.year <= pt_train_cfg["end_year"])
    mask_pt_val = (dates_era5.year >= pt_val_cfg["start_year"]) & (dates_era5.year <= pt_val_cfg["end_year"])
    
    print(f"ERA5 pre-train train dates: {pt_train_cfg['start_year']}-{pt_train_cfg['end_year']} ({mask_pt_train.sum()} days)")
    print(f"ERA5 pre-train val dates:   {pt_val_cfg['start_year']}-{pt_val_cfg['end_year']} ({mask_pt_val.sum()} days)")

    # 3. Fit Normalization Statistics (z-score) STRICTLY ON ERA5 TRAIN ONLY
    print("Fitting normalization statistics (mean and std) exclusively on ERA5 pre-training training split...")
    era5_train_data = era5_stacked[mask_pt_train]  # shape: (n_train_days, 6, 6, 15)
    
    # Reshape to (N, 15)
    train_flat = era5_train_data.reshape(-1, len(all_15_names))
    means = np.mean(train_flat, axis=0)
    stds = np.std(train_flat, axis=0)
    stds = np.where(stds == 0, 1.0, stds)

    scaler_stats = {
        "features": all_15_names,
        "mean": means.tolist(),
        "std": stds.tolist(),
    }
    scaler_path = os.path.join(processed_dir, "scaler_stats.json")
    with open(scaler_path, "w") as f:
        json.dump(scaler_stats, f, indent=2)
    print(f"Saved frozen scaler statistics to {scaler_path}")

    # Standardize ERA5 stacked predictors
    era5_norm = (era5_stacked - means) / stds

    # 4. Target transform: y' = ln(1 + y)
    era5_tp = predictor_arrays["tp"]
    era5_target_transformed = np.log1p(era5_tp)

    imd_rain = np.maximum(ds_imd.rain.values, 0.0)
    imd_target_transformed = np.log1p(imd_rain)

    # 5. Compute Real Class Distribution on IMD Fine-Tuning Pool (2000-2015)
    ft_pool_cfg = cfg["partitions"]["imd_finetune_pool"]
    mask_ft_pool = (dates_imd.year >= ft_pool_cfg["start_year"]) & (dates_imd.year <= ft_pool_cfg["end_year"])
    imd_pool_rain = imd_rain[mask_ft_pool]

    thresholds = cfg["rainfall_classes"]["thresholds"]
    class_dist = compute_class_distribution(imd_pool_rain, thresholds)
    dist_path = os.path.join(processed_dir, "class_distribution.json")
    with open(dist_path, "w") as f:
        json.dump(class_dist, f, indent=2)
    print(f"Real class shares on IMD fine-tuning pool (2000-2015):\n{json.dumps(class_dist['shares'], indent=2)}")

    # 6. Organize cell-by-cell data for all partitions
    # We flatten (lat, lon) to 36 cells: cell index c in [0, 35]
    print("Formatting cell-by-cell time series across partitions...")
    
    def extract_cell_series(arr_3d_or_4d):
        # returns list of 36 arrays
        cells = []
        for i_lat in range(len(lats)):
            for i_lon in range(len(lons)):
                cells.append(arr_3d_or_4d[:, i_lat, i_lon])
        return cells

    # Match ERA5 dates for IMD periods
    imd_val_cfg = cfg["partitions"]["imd_validation"]
    imd_test_cfg = cfg["partitions"]["imd_test"]

    mask_imd_val = (dates_imd.year >= imd_val_cfg["start_year"]) & (dates_imd.year <= imd_val_cfg["end_year"])
    mask_imd_test = (dates_imd.year >= imd_test_cfg["start_year"]) & (dates_imd.year <= imd_test_cfg["end_year"])
    
    # In ERA5:
    mask_era5_imd_pool = (dates_era5.year >= ft_pool_cfg["start_year"]) & (dates_era5.year <= ft_pool_cfg["end_year"])
    mask_era5_imd_val = (dates_era5.year >= imd_val_cfg["start_year"]) & (dates_era5.year <= imd_val_cfg["end_year"])
    mask_era5_imd_test = (dates_era5.year >= imd_test_cfg["start_year"]) & (dates_era5.year <= imd_test_cfg["end_year"])

    np.savez_compressed(
        os.path.join(processed_dir, "pretrain_data.npz"),
        era5_train_X=extract_cell_series(era5_norm[mask_pt_train]),
        era5_train_y=extract_cell_series(era5_target_transformed[mask_pt_train]),
        era5_val_X=extract_cell_series(era5_norm[mask_pt_val]),
        era5_val_y=extract_cell_series(era5_target_transformed[mask_pt_val]),
        feature_names=all_15_names,
    )

    np.savez_compressed(
        os.path.join(processed_dir, "finetune_data.npz"),
        imd_pool_X=extract_cell_series(era5_norm[mask_era5_imd_pool]),
        imd_pool_y=extract_cell_series(imd_target_transformed[mask_ft_pool]),
        imd_pool_y_raw=extract_cell_series(imd_rain[mask_ft_pool]),
        imd_val_X=extract_cell_series(era5_norm[mask_era5_imd_val]),
        imd_val_y=extract_cell_series(imd_target_transformed[mask_imd_val]),
        imd_val_y_raw=extract_cell_series(imd_rain[mask_imd_val]),
        imd_test_X=extract_cell_series(era5_norm[mask_era5_imd_test]),
        imd_test_y=extract_cell_series(imd_target_transformed[mask_imd_test]),
        imd_test_y_raw=extract_cell_series(imd_rain[mask_imd_test]),
        dates_test=[str(d.date()) for d in dates_imd[mask_imd_test]],
        dates_val=[str(d.date()) for d in dates_imd[mask_imd_val]],
        dates_pool=[str(d.date()) for d in dates_imd[mask_ft_pool]],
    )
    print("Preprocessing completed successfully. Processed artifacts saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    preprocess_all(fast=args.fast)
