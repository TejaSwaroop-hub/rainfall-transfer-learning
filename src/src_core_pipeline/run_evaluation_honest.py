"""
Honest Core Evaluation Script -- replaces run_experiments.py fabrications.

Runs ONLY experiments for which checkpoints or analytical solutions exist:
  M0: Persistence baseline (no model needed)
  M1: Day-of-year Climatology (training data only)
  M2: From-Scratch GRU, seed 0, N=5 yr  (checkpoint: theta_scratch_gru_N5_seed0.pt)
  M3: From-Scratch LSTM, seed 0, N=5 yr (checkpoint: theta_scratch_lstm_N5_seed0.pt)
  M4: Zero-Shot ERA5 GRU (checkpoint: theta_pre_gru_seed0.pt evaluated on IMD test)
  M6: ERA5->IMD Transfer GRU S1, seed 0, N=5 yr (theta_ft_gru_N5_S1_seed0.pt)

Saves:
  results/predictions_verified.csv
  results/metrics_honest.json
  tables/table1_honest.md
  tables/table2_honest.md
"""

import os, sys, json, csv
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.models import RainfallRecurrentModel
from src.evaluate import compute_continuous_metrics, compute_contingency_metrics, compute_wilcoxon_test
from src.windows import create_sliding_windows_for_cell

import yaml
CFG_PATH = os.path.join(ROOT, "configs", "default.yaml")
with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

LOOKBACK   = cfg["model"]["lookback_window"]
HIDDEN_DIM = cfg["model"]["hidden_dim"]
NUM_LAYERS = cfg["model"]["num_layers"]
DROPOUT    = cfg["model"]["dropout"]
INPUT_DIM  = 10

THRESHOLDS = [2.5, 15.6, 64.5]
DEVICE     = torch.device("cpu")

# ── Load processed data (arrays are shape (36, N_days, 15) or (36, N_days)) ──
print("Loading finetune_data.npz ...")
proc_dir = os.path.join(ROOT, "data", "processed")
ft = np.load(os.path.join(proc_dir, "finetune_data.npz"), allow_pickle=True)

# Shape: (36, 1826, 15)  and  (36, 1826)
imd_test_X     = ft["imd_test_X"]      # (36, 1826, 15)
imd_test_y_raw = ft["imd_test_y_raw"]  # (36, 1826) mm
imd_test_y     = ft["imd_test_y"]      # (36, 1826) log-transformed
imd_pool_y_raw = ft["imd_pool_y_raw"]  # (36, 5844) mm
dates_test = list(ft["dates_test"])     # 1826 date strings
dates_pool = list(ft["dates_pool"])     # 5844 date strings

# Load feature index (top-10)
with open(os.path.join(proc_dir, "selected_features.json")) as f:
    feat_info = json.load(f)
pt_data = np.load(os.path.join(proc_dir, "pretrain_data.npz"), allow_pickle=True)
all_names    = list(pt_data["feature_names"])
selected_idx = [all_names.index(name) for name in feat_info["selected_features"]]
assert len(selected_idx) == 10, f"Expected 10 features, got {len(selected_idx)}"

n_cells     = imd_test_X.shape[0]   # 36
n_test_days = imd_test_X.shape[1]   # 1826
print(f"  Cells: {n_cells}, Test days: {n_test_days}, Lookback: {LOOKBACK}")

# Filter to selected features -> (36, 1826, 10)
test_X_filt = imd_test_X[:, :, selected_idx]

# ── Build windows per cell ────────────────────────────────────────────────
print("Building sliding windows per cell ...")
cell_data = []  # each entry: (X_c, y_obs_c, cdates)
for ci in range(n_cells):
    X_c, _ = create_sliding_windows_for_cell(
        test_X_filt[ci], imd_test_y[ci], lookback=LOOKBACK, horizon=1)
    # Also get raw mm targets aligned to same windows
    # The windows skip NaN positions; for simplicity use direct slicing
    # (no NaNs in this dataset per audit)
    n_w = n_test_days - LOOKBACK   # number of valid windows
    y_obs = imd_test_y_raw[ci, LOOKBACK:]   # shape (n_w,)
    cdates = dates_test[LOOKBACK:]           # list of n_w dates
    cell_data.append((X_c, y_obs.astype(np.float32), cdates))

total_windows = sum(len(cd[0]) for cd in cell_data)
print(f"  Total test windows: {total_windows}")

# ── Load a checkpoint ────────────────────────────────────────────────────
def load_model(ckpt_name, cell_type="GRU"):
    path = os.path.join(ROOT, "results", "models", ckpt_name)
    if not os.path.exists(path):
        print(f"    MISSING: {path}")
        return None
    model = RainfallRecurrentModel(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        cell_type=cell_type,
    )
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"    Loaded: {ckpt_name}")
    return model

def predict_model(model, X_cell):
    """X_cell: (N, L, K) -> predictions in mm."""
    if len(X_cell) == 0:
        return np.array([], dtype=np.float32)
    with torch.no_grad():
        X_t = torch.tensor(X_cell, dtype=torch.float32)
        logpred = model(X_t).squeeze(-1).numpy()
    return np.maximum(np.expm1(logpred), 0.0)

# ── Baselines ─────────────────────────────────────────────────────────────
def persistence_preds(ci):
    """M0: forecast = last observed rainfall (day t predicts day t+1)."""
    raw = imd_test_y_raw[ci]  # (1826,)
    # Window at position t uses days [t-LOOKBACK+1 .. t], predicts t+1
    # So last observed in lookback is raw[t], predicting raw[t+1] = raw[LOOKBACK..]
    return raw[LOOKBACK - 1: -1].astype(np.float32)   # shape (1812,) = 1826-14

def climatology_preds(ci, cdates):
    """M1: DOY climatology from pool (all 36 cells pooled)."""
    doy_sum = {}
    doy_cnt = {}
    for ci2 in range(n_cells):
        pool_cell = imd_pool_y_raw[ci2]  # (5844,)
        for di, ds in enumerate(dates_pool):
            doy = int(ds[5:7]) * 100 + int(ds[8:10])
            doy_sum[doy] = doy_sum.get(doy, 0.0) + float(pool_cell[di])
            doy_cnt[doy] = doy_cnt.get(doy, 0) + 1
    doy_mean = {d: doy_sum[d] / doy_cnt[d] for d in doy_sum}
    preds = []
    for d in cdates:
        doy = int(d[5:7]) * 100 + int(d[8:10])
        preds.append(doy_mean.get(doy, 0.0))
    return np.array(preds, dtype=np.float32)

# ── Build all model predictions ───────────────────────────────────────────
print("\nBuilding model predictions ...")
# Dict: model_id -> dict(cell_idx -> (y_obs, y_pred))
model_cell_preds = {}

# M0: Persistence
print("  M0: Persistence ...")
m0 = {}
for ci, (X_c, y_obs, cdates) in enumerate(cell_data):
    p = persistence_preds(ci)
    n = min(len(y_obs), len(p))
    m0[ci] = (y_obs[:n], p[:n])
model_cell_preds["M0_Persistence"] = m0

# M1: Climatology
print("  M1: Climatology ...")
m1 = {}
for ci, (X_c, y_obs, cdates) in enumerate(cell_data):
    p = climatology_preds(ci, cdates)
    n = min(len(y_obs), len(p))
    m1[ci] = (y_obs[:n], p[:n])
model_cell_preds["M1_Climatology"] = m1

# M2: Scratch GRU
print("  M2: Scratch GRU ...")
m2_model = load_model("theta_scratch_gru_N5_seed0.pt", "GRU")
if m2_model:
    m2 = {}
    for ci, (X_c, y_obs, cdates) in enumerate(cell_data):
        p = predict_model(m2_model, X_c)
        n = min(len(y_obs), len(p))
        m2[ci] = (y_obs[:n], p[:n])
    model_cell_preds["M2_Scratch_GRU"] = m2

# M3: Scratch LSTM
print("  M3: Scratch LSTM ...")
m3_model = load_model("theta_scratch_lstm_N5_seed0.pt", "LSTM")
if m3_model:
    m3 = {}
    for ci, (X_c, y_obs, cdates) in enumerate(cell_data):
        p = predict_model(m3_model, X_c)
        n = min(len(y_obs), len(p))
        m3[ci] = (y_obs[:n], p[:n])
    model_cell_preds["M3_Scratch_LSTM"] = m3

# M4: Zero-Shot
print("  M4: Zero-Shot ERA5->IMD ...")
m4_model = load_model("theta_pre_gru_seed0.pt", "GRU")
if m4_model:
    m4 = {}
    for ci, (X_c, y_obs, cdates) in enumerate(cell_data):
        p = predict_model(m4_model, X_c)
        n = min(len(y_obs), len(p))
        m4[ci] = (y_obs[:n], p[:n])
    model_cell_preds["M4_ZeroShot_GRU"] = m4

# M6: Transfer GRU S1
print("  M6: Transfer GRU (S1, N=5yr) ...")
m6_model = load_model("theta_ft_gru_N5_S1_seed0.pt", "GRU")
if m6_model:
    m6 = {}
    for ci, (X_c, y_obs, cdates) in enumerate(cell_data):
        p = predict_model(m6_model, X_c)
        n = min(len(y_obs), len(p))
        m6[ci] = (y_obs[:n], p[:n])
    model_cell_preds["M6_Transfer_GRU_S1"] = m6

# ── Save predictions CSV ──────────────────────────────────────────────────
csv_path = os.path.join(ROOT, "results", "predictions_verified.csv")
print(f"\nSaving {csv_path} ...")

import xarray as xr
ds_imd = xr.open_dataset(os.path.join(ROOT, "data", "raw", "imd", "imd_rain_2000_2023.nc"))
lats = ds_imd.lat.values
lons = ds_imd.lon.values
ds_imd.close()
cell_coords = [(float(lats[i//len(lons)]), float(lons[i%len(lons)])) for i in range(n_cells)]

with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
    writer = csv.writer(csvf)
    writer.writerow(["model_id","cell_idx","cell_lat","cell_lon",
                     "date","horizon_days","observed_mm","predicted_mm"])
    for model_id, cell_dict in model_cell_preds.items():
        for ci, (y_obs, y_pred) in cell_dict.items():
            clat, clon = cell_coords[ci]
            cdates = cell_data[ci][2]
            n = min(len(y_obs), len(y_pred), len(cdates))
            for i in range(n):
                writer.writerow([model_id, ci, f"{clat:.4f}", f"{clon:.4f}",
                                 cdates[i], 1,
                                 f"{float(y_obs[i]):.4f}", f"{float(y_pred[i]):.4f}"])
print(f"  Done.")

# ── Compute metrics ───────────────────────────────────────────────────────
print("\nComputing metrics ...")
results = {}
for model_id, cell_dict in model_cell_preds.items():
    all_obs  = np.concatenate([v[0] for v in cell_dict.values()])
    all_pred = np.concatenate([v[1] for v in cell_dict.values()])
    n = min(len(all_obs), len(all_pred))
    all_obs, all_pred = all_obs[:n], all_pred[:n]

    cont = compute_continuous_metrics(all_obs, all_pred)
    cat  = {}
    for thr in THRESHOLDS:
        c = compute_contingency_metrics(all_obs, all_pred, thr)
        cat[f"thr_{thr}mm"] = {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in c.items()}
    results[model_id] = {
        "n_samples": int(n),
        "continuous": {k: round(v, 4) for k, v in cont.items()},
        "contingency": cat,
    }
    c = results[model_id]["continuous"]
    print(f"  {model_id}: MAE={c['MAE']:.2f} RMSE={c['RMSE']:.2f} "
          f"NSE={c['NSE']:.3f} KGE={c['KGE']:.3f} PBias={c['PBias']:.1f}%")

# Wilcoxon: M2 vs M6 per-cell RMSE
if "M2_Scratch_GRU" in model_cell_preds and "M6_Transfer_GRU_S1" in model_cell_preds:
    rmse_m2, rmse_m6 = [], []
    for ci in range(n_cells):
        obs2, p2 = model_cell_preds["M2_Scratch_GRU"][ci]
        obs6, p6 = model_cell_preds["M6_Transfer_GRU_S1"][ci]
        n2 = min(len(obs2), len(p2))
        n6 = min(len(obs6), len(p6))
        if n2 > 0:
            rmse_m2.append(float(np.sqrt(np.mean((obs2[:n2] - p2[:n2])**2))))
        if n6 > 0:
            rmse_m6.append(float(np.sqrt(np.mean((obs6[:n6] - p6[:n6])**2))))
    if len(rmse_m2) == len(rmse_m6) and len(rmse_m2) > 1:
        W, p = compute_wilcoxon_test(np.array(rmse_m2), np.array(rmse_m6), alternative="less")
        results["wilcoxon_M6_vs_M2"] = {
            "W_statistic": round(float(W), 4),
            "p_value": round(float(p), 6),
            "n_cells": len(rmse_m2),
            "note": "COMPUTED from actual per-cell RMSEs (seed=0 only)"
        }
        print(f"\n  Wilcoxon (M6 RMSE < M2 RMSE): W={W:.1f}, p={p:.4f}")

metrics_path = os.path.join(ROOT, "results", "metrics_honest.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"Saved {metrics_path}")

# ── Table 1 ───────────────────────────────────────────────────────────────
MODEL_LABELS = {
    "M0_Persistence":     "M0 Persistence",
    "M1_Climatology":     "M1 Climatology",
    "M2_Scratch_GRU":     "M2 Scratch GRU (N=5yr, seed=0)",
    "M3_Scratch_LSTM":    "M3 Scratch LSTM (N=5yr, seed=0)",
    "M4_ZeroShot_GRU":    "M4 Zero-Shot GRU (ERA5 pretrained)",
    "M6_Transfer_GRU_S1": "M6 Transfer GRU S1 (ERA5->IMD, N=5yr, seed=0)",
}

tbl1 = [
    "# Table 1 -- Continuous Performance Metrics (Actual, Single Seed)",
    "",
    "> **Data provenance**: 36-cell IMD test set (2019-2023), 1,812 windows per cell.",
    "> Single run (seed=0). Metrics computed from actual checkpoint predictions.",
    "",
    "| Model | N samples | MAE (mm) | RMSE (mm) | PBias (%) | NSE | KGE |",
    "|-------|-----------|----------|-----------|-----------|-----|-----|",
]
for mid, label in MODEL_LABELS.items():
    if mid not in results:
        continue
    r = results[mid]; c = r["continuous"]
    tbl1.append(f"| {label} | {r['n_samples']:,} | {c['MAE']:.2f} | {c['RMSE']:.2f} | "
                f"{c['PBias']:.1f} | {c['NSE']:.3f} | {c['KGE']:.3f} |")
tbl1 += ["", "*Note: Multi-seed, N-sweep, ablation experiments are future work.*"]

tbl1_path = os.path.join(ROOT, "tables", "table1_honest.md")
with open(tbl1_path, "w", encoding="utf-8") as f:
    f.write("\n".join(tbl1))
print(f"Saved {tbl1_path}")

# ── Table 2 ───────────────────────────────────────────────────────────────
tbl2 = [
    "# Table 2 -- Contingency Skill Scores (Actual, Single Seed)",
    "",
    "> Thresholds follow IMD rainfall classification (2.5=light, 15.6=moderate, 64.5=heavy).",
    "",
]
for thr in THRESHOLDS:
    key = f"thr_{thr}mm"
    tbl2.append(f"## Threshold: {thr} mm/day")
    tbl2.append("")
    tbl2.append("| Model | POD | FAR | CSI | HSS | Hits | FA | Misses |")
    tbl2.append("|-------|-----|-----|-----|-----|------|----|--------|")
    for mid, label in MODEL_LABELS.items():
        if mid not in results or key not in results[mid]["contingency"]:
            continue
        c = results[mid]["contingency"][key]
        tbl2.append(f"| {label} | {c['POD']:.3f} | {c['FAR']:.3f} | {c['CSI']:.3f} | "
                    f"{c['HSS']:.3f} | {c['hits']} | {c['false_alarms']} | {c['misses']} |")
    tbl2.append("")

tbl2_path = os.path.join(ROOT, "tables", "table2_honest.md")
with open(tbl2_path, "w", encoding="utf-8") as f:
    f.write("\n".join(tbl2))
print(f"Saved {tbl2_path}")

print("\n=== Evaluation complete ===")
print(f"  predictions_verified.csv : {csv_path}")
print(f"  metrics_honest.json      : {metrics_path}")
print(f"  table1_honest.md         : {tbl1_path}")
print(f"  table2_honest.md         : {tbl2_path}")
