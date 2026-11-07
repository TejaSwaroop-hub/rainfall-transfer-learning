#!/usr/bin/env python3
"""Create publication-ready comparison figures from the consolidated 2023 metrics CSV.

Run from the project root:
    python src/create_final_comparison_figures.py

Input:
    results/metrics/final_master_model_comparison_2023.csv
Outputs:
    results/figures/final_2023_model_ranking_csi.png
    results/figures/final_2023_error_metrics.png
    results/figures/final_2023_skill_metrics.png
    results/figures/final_2023_event_detection_metrics.png
"""
from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "metrics" / "final_master_model_comparison_2023.csv"
OUT = ROOT / "results" / "figures"


def clean_label(name: str) -> str:
    return (name.replace("_", " ").title()
            .replace("Gru", "GRU").replace("Era5", "ERA5"))


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {INPUT}\n"
            "Run src/rebuild_final_master_comparison.py first."
        )

    df = pd.read_csv(INPUT)
    required = {
        "model", "MAE_mm", "RMSE_mm", "NSE", "KGE", "POD", "FAR", "CSI", "HSS"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(df) != 14:
        raise ValueError(f"Expected 14 models, found {len(df)}")

    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.titlesize": 14,
    })

    # Figure 1: CSI ranking (all 14 models).
    ranked = df.sort_values("CSI", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh([clean_label(x) for x in ranked["model"]], ranked["CSI"])
    ax.set_title("Rainfall Event Prediction: CSI Ranking (2023)")
    ax.set_xlabel("Critical Success Index (CSI)")
    ax.set_xlim(0, max(0.22, float(ranked["CSI"].max()) * 1.18))
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, ranked["CSI"]):
        ax.text(value + 0.002, bar.get_y() + bar.get_height()/2,
                f"{value:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "final_2023_model_ranking_csi.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: amount-error metrics, separated into panels for readability.
    # Lower errors are better; descending sort puts the best (lowest) model at the top.
    by_mae = df.sort_values("MAE_mm", ascending=False).copy()
    by_rmse = df.sort_values("RMSE_mm", ascending=False).copy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, data, column, title in [
        (axes[0], by_mae, "MAE_mm", "Mean Absolute Error (MAE)"),
        (axes[1], by_rmse, "RMSE_mm", "Root Mean Squared Error (RMSE)"),
    ]:
        bars = ax.barh([clean_label(x) for x in data["model"]], data[column])
        ax.set_title(title)
        ax.set_xlabel("Rainfall error (mm/day)")
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, data[column]):
            ax.text(value + 0.035, bar.get_y() + bar.get_height()/2,
                    f"{value:.2f}", va="center", fontsize=7)
        ax.set_xlim(0, float(data[column].max()) * 1.12)
    fig.suptitle("Rainfall Amount Prediction Errors (2023)")
    fig.tight_layout()
    fig.savefig(OUT / "final_2023_error_metrics.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 3: NSE and KGE, with the zero-skill reference line.
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, column, title in [
        (axes[0], "NSE", "Nash–Sutcliffe Efficiency (NSE)"),
        (axes[1], "KGE", "Kling–Gupta Efficiency (KGE)"),
    ]:
        data = df.dropna(subset=[column]).sort_values(column, ascending=True)
        bars = ax.barh([clean_label(x) for x in data["model"]], data[column])
        ax.axvline(0, linewidth=1, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Efficiency score (higher is better)")
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, data[column]):
            offset = 0.012 if value >= 0 else -0.012
            align = "left" if value >= 0 else "right"
            ax.text(value + offset, bar.get_y() + bar.get_height()/2,
                    f"{value:.3f}", va="center", ha=align, fontsize=7)
        # Add horizontal padding so annotations on the most negative bars stay inside the axes.
        low = min(0.0, float(data[column].min()))
        high = max(0.0, float(data[column].max()))
        span = max(high - low, 0.1)
        pad = 0.12 * span
        ax.set_xlim(low - pad, high + pad)
    fig.suptitle("Rainfall Prediction Skill Metrics (2023)")
    fig.tight_layout()
    fig.savefig(OUT / "final_2023_skill_metrics.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 4: event-detection measures, with FAR shown as a separate panel.
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, column, title in [
        (axes[0], "POD", "Probability of Detection (POD)"),
        (axes[1], "FAR", "False Alarm Ratio (FAR)"),
        (axes[2], "HSS", "Heidke Skill Score (HSS)"),
    ]:
        # POD and HSS: higher is better. FAR: lower is better.
        ascending = (column != "FAR")
        data = df.dropna(subset=[column]).sort_values(column, ascending=ascending)
        bars = ax.barh([clean_label(x) for x in data["model"]], data[column])
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Score")
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, data[column]):
            ax.text(min(value + 0.015, 0.96), bar.get_y() + bar.get_height()/2,
                    f"{value:.2f}", va="center", fontsize=7)
    fig.suptitle("Rainfall Event Detection Metrics (2023; event threshold = 2.5 mm/day)")
    fig.tight_layout()
    fig.savefig(OUT / "final_2023_event_detection_metrics.png", bbox_inches="tight")
    plt.close(fig)

    outputs = [
        "final_2023_model_ranking_csi.png",
        "final_2023_error_metrics.png",
        "final_2023_skill_metrics.png",
        "final_2023_event_detection_metrics.png",
    ]
    print("FINAL COMPARISON FIGURES CREATED")
    for name in outputs:
        path = OUT / name
        print(f"{path.relative_to(ROOT)} | {path.stat().st_size:,} bytes")
    print("All four figures use the 14 models in the final master comparison CSV.")
    print("Note: the always-zero model has undefined FAR and KGE; those entries are omitted.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
