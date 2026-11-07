"""
make_figures_honest.py — Generates all figures from ACTUAL computed results.

Reads from results/predictions_verified.csv and results/metrics_honest.json.
Requires: numpy, matplotlib, seaborn, scipy, torch (for weight-shift figure).

Figures:
  fig1 — Domain map (legitimate: grid coordinates are real)
  fig2 — Contingency score bars (from actual predictions)
  fig5 — Hyetograph / event trace for a selected 80-day storm window (actual predictions)
  fig6 — Layer-wise weight shift (computed from actual checkpoint files)

Figs 3 (sample efficiency sweep) and 4 (lead-time degradation) are removed.
They require experiments across N∈{1,2,3,5,10,16} years and h∈{1,3,7} horizons
that were NOT run. They are listed as future work in the report.
"""

import os, sys, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

FIG_DIR = os.path.join(ROOT, "figures")
RESULTS_DIR = os.path.join(ROOT, "results")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics_honest.json")
PREDS_PATH   = os.path.join(RESULTS_DIR, "predictions_verified.csv")
PROC_DIR     = os.path.join(ROOT, "data", "processed")

os.makedirs(FIG_DIR, exist_ok=True)

# ── Load computed results ─────────────────────────────────────────────────
if not os.path.exists(METRICS_PATH):
    raise FileNotFoundError(f"Run src/run_evaluation_honest.py first to generate {METRICS_PATH}")
with open(METRICS_PATH) as f:
    metrics = json.load(f)

# Load predictions CSV
preds_by_model = {}
if os.path.exists(PREDS_PATH):
    with open(PREDS_PATH, newline="", encoding="utf-8") as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            mid = row["model_id"]
            if mid not in preds_by_model:
                preds_by_model[mid] = {"date": [], "obs": [], "pred": [], "cell": []}
            preds_by_model[mid]["date"].append(row["date"])
            preds_by_model[mid]["obs"].append(float(row["observed_mm"]))
            preds_by_model[mid]["pred"].append(float(row["predicted_mm"]))
            preds_by_model[mid]["cell"].append(int(row["cell_idx"]))

MODEL_LABELS = {
    "M0_Persistence":     "M0 Persistence",
    "M1_Climatology":     "M1 Climatology",
    "M2_Scratch_GRU":     "M2 Scratch GRU",
    "M3_Scratch_LSTM":    "M3 Scratch LSTM",
    "M4_ZeroShot_GRU":    "M4 Zero-Shot GRU",
    "M6_Transfer_GRU_S1": "M6 Transfer GRU (S1)",
}
COLORS = {
    "M0_Persistence":     "#aaaaaa",
    "M1_Climatology":     "#cccccc",
    "M2_Scratch_GRU":     "#4878d0",
    "M3_Scratch_LSTM":    "#6acc65",
    "M4_ZeroShot_GRU":    "#ee854a",
    "M6_Transfer_GRU_S1": "#d65f5f",
}

# ── Fig 1: Domain map ─────────────────────────────────────────────────────
print("Plotting Fig 1: Domain map ...")
# Cell coordinates from IMD NetCDF
import xarray as xr
ds_imd = xr.open_dataset(os.path.join(ROOT, "data", "raw", "imd", "imd_rain_2000_2023.nc"))
lats = ds_imd.lat.values
lons = ds_imd.lon.values
ds_imd.close()

fig, ax = plt.subplots(figsize=(6, 5))
lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
ax.scatter(lon_grid.ravel(), lat_grid.ravel(), s=80, c="#2166ac",
           marker="s", zorder=3, label=f"{len(lats)}×{len(lons)} IMD grid cells")
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title("Study Domain: Anantapur District, AP\n(IMD 1°×1° grid cells)")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
# Approximate Anantapur district outline
region_lat = [min(lats)-0.1, min(lats)-0.1, max(lats)+0.1, max(lats)+0.1, min(lats)-0.1]
region_lon = [min(lons)-0.1, max(lons)+0.1, max(lons)+0.1, min(lons)-0.1, min(lons)-0.1]
ax.plot(region_lon, region_lat, "r--", linewidth=1.5, label="Study region boundary")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig1_anantapur_domain_grid.png"), dpi=150)
plt.close()
print("  Saved fig1")

# ── Fig 2: Contingency skill scores ──────────────────────────────────────
print("Plotting Fig 2: Contingency skill scores ...")
THRESHOLDS = [2.5, 15.6, 64.5]
THR_LABELS = ["Light (≥2.5 mm)", "Moderate (≥15.6 mm)", "Heavy (≥64.5 mm)"]
SCORE_TYPES = ["POD", "CSI", "HSS"]
SCORE_FULL  = ["Probability of Detection", "Critical Success Index", "Heidke Skill Score"]

mids_ordered = [m for m in MODEL_LABELS if m in metrics]
if mids_ordered:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax_i, (thr, thr_label) in enumerate(zip(THRESHOLDS, THR_LABELS)):
        ax = axes[ax_i]
        thr_key = f"thr_{thr}mm"
        bars_data = {sc: [] for sc in SCORE_TYPES}
        bar_labels = []
        for mid in mids_ordered:
            if thr_key not in metrics.get(mid, {}).get("contingency", {}):
                continue
            cat = metrics[mid]["contingency"][thr_key]
            bar_labels.append(MODEL_LABELS[mid])
            for sc in SCORE_TYPES:
                bars_data[sc].append(cat.get(sc, 0.0))

        x = np.arange(len(bar_labels))
        width = 0.25
        for si, sc in enumerate(SCORE_TYPES):
            vals = bars_data[sc]
            if vals:
                ax.bar(x + si * width, vals, width, label=SCORE_FULL[si], alpha=0.85)

        ax.set_xticks(x + width)
        ax.set_xticklabels(bar_labels, rotation=35, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(thr_label)
        ax.legend(fontsize=7, loc="upper right")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Contingency Skill Scores — Actual Model Predictions\n"
                 "(Single seed, seed=0; 36-cell IMD test set 2019–2023)", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_contingency_scores.png"), dpi=150)
    plt.close()
    print("  Saved fig2")

# ── Fig 5: Hyetograph — actual predictions on a 80-day storm window ───────
print("Plotting Fig 5: Hyetograph (actual predictions) ...")
# Pick cell 0, find a 80-day monsoon window with high rainfall
if "M6_Transfer_GRU_S1" in preds_by_model and "M2_Scratch_GRU" in preds_by_model:
    m6d = preds_by_model["M6_Transfer_GRU_S1"]
    m2d = preds_by_model["M2_Scratch_GRU"]
    # Filter to cell 0
    idx_m6 = [i for i, c in enumerate(m6d["cell"]) if c == 0]
    idx_m2 = [i for i, c in enumerate(m2d["cell"]) if c == 0]

    obs  = np.array([m6d["obs"][i]  for i in idx_m6])
    pred_m6 = np.array([m6d["pred"][i] for i in idx_m6])
    pred_m2 = np.array([m2d["pred"][i] for i in idx_m2[:len(pred_m6)]])
    dates = [m6d["date"][i] for i in idx_m6]

    # Find an 80-day window with highest total rainfall
    W = 80
    best_start = 0
    best_total = -1
    for s in range(len(obs) - W):
        t = obs[s:s+W].sum()
        if t > best_total:
            best_total = t
            best_start = s

    obs_w    = obs[best_start: best_start + W]
    pred_m6_w = pred_m6[best_start: best_start + W]
    pred_m2_w = pred_m2[best_start: best_start + W] if len(pred_m2) >= best_start + W else np.zeros(W)
    t_axis = np.arange(W)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(t_axis, obs_w, color="#2166ac", alpha=0.5, label="Observed (IMD)", width=0.9)
    ax.plot(t_axis, pred_m6_w, color="#d65f5f", linewidth=1.8, label="M6 Transfer GRU (S1)")
    ax.plot(t_axis, pred_m2_w, color="#4878d0", linewidth=1.4, linestyle="--", label="M2 Scratch GRU")
    ax.set_xlabel(f"Day offset from {dates[best_start]}")
    ax.set_ylabel("Rainfall (mm/day)")
    ax.set_title(f"Fig 5: 80-day Storm Window — Cell 0 ({dates[best_start]} to {dates[best_start+W-1]})\n"
                 "Actual predictions from loaded checkpoints (seed=0)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_hyetographs_actual.png"), dpi=150)
    plt.close()
    print("  Saved fig5_hyetographs_actual.png")
else:
    print("  SKIPPED fig5: M6 or M2 predictions not available. Run run_evaluation_honest.py first.")

# ── Fig 6: Weight shift from actual checkpoints ───────────────────────────
print("Plotting Fig 6: Weight shift (from actual checkpoints) ...")
try:
    import torch
    sys.path.insert(0, ROOT)
    from src.interpret import compute_layer_weight_shift
    pre_path = os.path.join(ROOT, "results", "models", "theta_pre_gru_seed0.pt")
    ft_path  = os.path.join(ROOT, "results", "models", "theta_ft_gru_N5_S1_seed0.pt")
    if os.path.exists(pre_path) and os.path.exists(ft_path):
        ws = compute_layer_weight_shift(pre_path, ft_path)
        layers = ["GRU Layer 1\n(lower-level)", "GRU Layer 2\n(higher-level)", "Dense Head\n(output layer)"]
        deltas = [ws["delta_layer1"] * 100, ws["delta_layer2"] * 100, ws["delta_head"] * 100]
        fig, ax = plt.subplots(figsize=(6, 4))
        bar_colors = ["#4878d0", "#6acc65", "#d65f5f"]
        bars = ax.bar(layers, deltas, color=bar_colors, alpha=0.85, edgecolor="k", linewidth=0.7)
        for bar, val in zip(bars, deltas):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylabel("Relative Weight Shift ΔFrob (%)")
        ax.set_title("Fig 6: Layer-wise Relative Weight Shift\n"
                     "(ERA5 pretrained → IMD fine-tuned, Strategy S1, seed=0)\n"
                     "Computed: ||θ_ft − θ_pre||₂ / ||θ_pre||₂")
        ax.set_ylim(0, max(deltas) * 1.3 + 1)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig6_weight_shift_actual.png"), dpi=150)
        plt.close()
        # Save the numbers
        ws_path = os.path.join(ROOT, "results", "weight_shift_honest.json")
        with open(ws_path, "w") as wf:
            json.dump({k: round(v, 6) for k, v in ws.items()}, wf, indent=2)
        print(f"  Saved fig6_weight_shift_actual.png and {ws_path}")
    else:
        print("  SKIPPED fig6: checkpoint files not found.")
except Exception as e:
    print(f"  ERROR in fig6: {e}")

# ── Scatter plot: Observed vs Predicted (bonus, helpful for defense) ───────
print("Plotting bonus scatter: Observed vs Predicted ...")
mids_to_plot = [m for m in ["M2_Scratch_GRU", "M6_Transfer_GRU_S1"] if m in preds_by_model]
if mids_to_plot:
    fig, axes = plt.subplots(1, len(mids_to_plot), figsize=(6 * len(mids_to_plot), 5))
    if len(mids_to_plot) == 1:
        axes = [axes]
    for ax, mid in zip(axes, mids_to_plot):
        d = preds_by_model[mid]
        obs_all  = np.array(d["obs"])
        pred_all = np.array(d["pred"])
        ax.scatter(obs_all, pred_all, s=4, alpha=0.3, color=COLORS.get(mid, "#333333"))
        maxv = max(obs_all.max(), pred_all.max(), 1)
        ax.plot([0, maxv], [0, maxv], "k--", linewidth=1, label="Perfect forecast")
        cont = metrics.get(mid, {}).get("continuous", {})
        ax.set_xlabel("Observed (mm/day)")
        ax.set_ylabel("Predicted (mm/day)")
        ax.set_title(f"{MODEL_LABELS.get(mid, mid)}\n"
                     f"MAE={cont.get('MAE','?'):.2f} RMSE={cont.get('RMSE','?'):.2f} "
                     f"NSE={cont.get('NSE','?'):.3f}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Observed vs. Predicted Rainfall — 36-cell IMD Test Set (2019–2023)", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_scatter_obs_vs_pred.png"), dpi=150)
    plt.close()
    print("  Saved fig_scatter_obs_vs_pred.png")

print("\n=== Figure generation complete ===")
print(f"  Figures saved to: {FIG_DIR}")
