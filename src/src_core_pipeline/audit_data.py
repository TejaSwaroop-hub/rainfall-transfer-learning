"""
Data Authenticity Audit Script.
Inspects ERA5 and IMD NetCDF files to verify they are real (not synthetic).
Writes findings to audit_findings.txt in the project root.
"""

import os, sys, json
import numpy as np
import xarray as xr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ERA5_PATH = os.path.join(ROOT, "data", "raw", "era5", "era5_anantapur_1979_2023.nc")
IMD_PATH  = os.path.join(ROOT, "data", "raw", "imd",  "imd_rain_2000_2023.nc")
OUT_PATH  = os.path.join(ROOT, "audit_findings.txt")

lines = []
def log(s=""):
    print(s)
    lines.append(s)

log("=" * 70)
log("DATA AUTHENTICITY AUDIT")
log("=" * 70)

# ── ERA5 ──────────────────────────────────────────────────────────────────
log("\n[1] ERA5 file: " + ERA5_PATH)
log(f"    File size: {os.path.getsize(ERA5_PATH)/1e6:.1f} MB")
ds5 = xr.open_dataset(ERA5_PATH)
log(f"    Variables : {list(ds5.data_vars)}")
log(f"    Dimensions: {dict(ds5.dims)}")
log(f"    Time range: {str(ds5.time.values[0])[:10]}  →  {str(ds5.time.values[-1])[:10]}")
log(f"    n_times   : {len(ds5.time)}")
log(f"    Coords    : {list(ds5.coords)}")

for v in ds5.data_vars:
    arr = ds5[v].values
    log(f"\n    Variable: {v}")
    log(f"      shape : {arr.shape}")
    log(f"      dtype : {arr.dtype}")
    log(f"      min   : {np.nanmin(arr):.4f}")
    log(f"      max   : {np.nanmax(arr):.4f}")
    log(f"      mean  : {np.nanmean(arr):.4f}")
    log(f"      std   : {np.nanstd(arr):.4f}")
    log(f"      NaNs  : {np.isnan(arr).sum()}")
    # Check if all spatial cells are identical (synthetic red flag)
    if arr.ndim >= 3:
        # shape is (time, lat, lon) or similar
        flat = arr.reshape(arr.shape[0], -1)  # (T, cells)
        cell_stds = np.nanstd(flat, axis=0)
        log(f"      spatial cell std range: [{cell_stds.min():.4f}, {cell_stds.max():.4f}]")
        # Inter-cell correlation: if all cells identical, std across cells = 0
        cross_cell_std = np.nanstd(flat, axis=1)   # std across cells at each time step
        log(f"      inter-cell std (mean over time): {np.mean(cross_cell_std):.4f}")
        # Check synthetic seasonal formula for tcwv
        if v == "tcwv":
            log("      [tcwv] Checking if tcwv matches a pure seasonal cosine formula...")
            # ERA5 tcwv at cell [0,0]
            cell00 = flat[:, 0]
            # Reconstruct what a perfect seasonal formula would give
            T = len(cell00)
            doy = np.arange(T) % 365
            baseline_amp = (cell00.max() - cell00.min()) / 2
            baseline_mid = (cell00.max() + cell00.min()) / 2
            synthetic = baseline_mid - baseline_amp * np.cos(2 * np.pi * doy / 365)
            resid = cell00 - synthetic
            corr = np.corrcoef(cell00, synthetic)[0, 1]
            log(f"      Pearson r(tcwv_cell00, pure_seasonal_cosine) = {corr:.4f}")
            log(f"      Residual std after removing seasonal signal    = {resid.std():.4f}")
            log(f"      If r≈1 AND residual≈0, data is synthetic. If r<0.95 or residual>1, data is real.")
            # Cross-cell identical check
            cell01 = flat[:, 1] if flat.shape[1] > 1 else cell00
            diff_cells = cell00 - cell01
            log(f"      Max |cell[0,0] - cell[0,1]|: {np.abs(diff_cells).max():.6f}")
            log(f"      If always 0.0 → all cells are identical (synthetic).")
ds5.close()

# ── IMD ──────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("[2] IMD file: " + IMD_PATH)
log(f"    File size: {os.path.getsize(IMD_PATH)/1e6:.1f} MB")
ds_imd = xr.open_dataset(IMD_PATH)
log(f"    Variables : {list(ds_imd.data_vars)}")
log(f"    Dimensions: {dict(ds_imd.dims)}")
log(f"    Time range: {str(ds_imd.time.values[0])[:10]}  →  {str(ds_imd.time.values[-1])[:10]}")
log(f"    n_times   : {len(ds_imd.time)}")
log(f"    Coords    : {list(ds_imd.coords)}")
for v in ds_imd.data_vars:
    arr = ds_imd[v].values
    log(f"\n    Variable: {v}")
    log(f"      shape : {arr.shape}")
    log(f"      min   : {np.nanmin(arr):.4f}")
    log(f"      max   : {np.nanmax(arr):.4f}")
    log(f"      mean  : {np.nanmean(arr):.4f}")
    log(f"      NaNs  : {np.isnan(arr).sum()}")
    # Rainfall class distribution
    flat_rain = arr[~np.isnan(arr)]
    if v.lower() in ("rain", "rf", "precipitation", "tp", "pr"):
        log(f"      % dry days (< 2.5 mm)       : {100*(flat_rain < 2.5).mean():.1f}%")
        log(f"      % light rain (2.5-15.6)     : {100*((flat_rain>=2.5)&(flat_rain<15.6)).mean():.1f}%")
        log(f"      % moderate rain (15.6-64.5) : {100*((flat_rain>=15.6)&(flat_rain<64.5)).mean():.1f}%")
        log(f"      % heavy rain (>=64.5 mm)    : {100*(flat_rain>=64.5).mean():.1f}%")
        log(f"      max single day rainfall     : {flat_rain.max():.1f} mm")
ds_imd.close()

# ── NPZ Files ────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("[3] Processed NPZ files")
for fname in ["finetune_data.npz", "pretrain_data.npz"]:
    fpath = os.path.join(ROOT, "data", "processed", fname)
    if os.path.exists(fpath):
        d = np.load(fpath, allow_pickle=True)
        log(f"\n  {fname}: keys = {list(d.keys())}")
        for k in d.keys():
            v = d[k]
            if hasattr(v, '__len__') and len(v) > 0:
                try:
                    arr = v[0] if isinstance(v[0], np.ndarray) else np.array(v)
                    if isinstance(v[0], np.ndarray):
                        log(f"    {k}: list of {len(v)} arrays, first shape={v[0].shape}, dtype={v[0].dtype}")
                    else:
                        log(f"    {k}: {type(v).__name__}, length={len(v)}, first={str(v[0])[:60]}")
                except Exception as e:
                    log(f"    {k}: {type(v).__name__}, length={len(v)} (error: {e})")
        d.close()

# ── Checkpoint files ──────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("[4] Checkpoint files")
import torch
ckpt_dir = os.path.join(ROOT, "results", "models")
for fname in os.listdir(ckpt_dir):
    fpath = os.path.join(ckpt_dir, fname)
    try:
        sd = torch.load(fpath, map_location="cpu", weights_only=True)
        keys = list(sd.keys())
        total_params = sum(v.numel() for v in sd.values())
        log(f"  {fname}: {len(keys)} keys, {total_params:,} total params")
        log(f"    layers: {keys[:4]}...")
    except Exception as e:
        log(f"  {fname}: ERROR loading — {e}")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

log(f"\nAudit written to: {OUT_PATH}")
