#!/usr/bin/env python3
"""Rebuild a complete 14-row exploratory 2023 model comparison.

Reads only existing metric CSVs; does not retrain models or alter predictions.
Ensures event threshold and contingency counts are retained for every model.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).parent.name == "src" else Path.cwd()
METRICS = ROOT / "results" / "metrics"
OUTPUT = METRICS / "final_master_model_comparison_2023.csv"
EVENT_THRESHOLD = 2.5
N_EXPECTED = 13_140

REQUIRED = [
    "model", "n_samples", "MAE_mm", "RMSE_mm", "NSE", "KGE",
    "event_threshold_mm", "POD", "FAR", "CSI", "HSS",
    "TP", "FP", "FN", "TN", "evaluation_status",
]


def read_csv(name: str) -> pd.DataFrame:
    path = METRICS / name
    if not path.exists():
        raise FileNotFoundError(f"Required metrics file not found: {path}")
    return pd.read_csv(path)


def normalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    if "event_threshold_mm" not in df.columns:
        df["event_threshold_mm"] = EVENT_THRESHOLD
    else:
        df["event_threshold_mm"] = df["event_threshold_mm"].fillna(EVENT_THRESHOLD)
    for col in REQUIRED:
        if col not in df.columns:
            df[col] = pd.NA
    # All rows are exploratory because 2023 was repeatedly examined during development.
    df["evaluation_status"] = "2023 exploratory/post-hoc; not an untouched test set"
    # Make contingency counts numeric where present.
    for col in ("TP", "FP", "FN", "TN", "n_samples"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[REQUIRED]


def main() -> None:
    parts = []

    # Four original single-stage model/reference rows.
    original = read_csv("model_comparison_test_2023.csv")
    parts.append(normalize(original, "original"))

    # Two-stage GRUs; event classification threshold is 2.5 mm/day.
    two_stage = read_csv("two_stage_metrics_test_2023.csv")
    parts.append(normalize(two_stage, "two_stage"))

    # Keep only the three spatial GRU variants, excluding duplicate baselines.
    spatial = read_csv("spatial_gru_metrics_2023_exploratory.csv")
    spatial = spatial[spatial["model"].isin([
        "spatial_zero_shot", "spatial_scratch", "spatial_transfer"
    ])]
    parts.append(normalize(spatial, "spatial"))

    # Two intensity-weighted GRUs.
    weighted = read_csv("intensity_weighted_gru_metrics_2023_exploratory.csv")
    parts.append(normalize(weighted, "weighted"))

    # Three additional simple baselines: monthly climatology, persistence, all-zero.
    baselines = read_csv("additional_baseline_comparison_2023.csv")
    parts.append(normalize(baselines, "additional_baselines"))

    master = pd.concat(parts, ignore_index=True)
    master["model"] = master["model"].astype(str)

    duplicates = master.loc[master["model"].duplicated(keep=False), "model"].tolist()
    if duplicates:
        raise ValueError(f"Duplicate model rows found: {sorted(set(duplicates))}")

    if len(master) != 14:
        raise ValueError(f"Expected 14 model/baseline rows, found {len(master)}")

    if not (master["n_samples"] == N_EXPECTED).all():
        bad = master.loc[master["n_samples"] != N_EXPECTED, ["model", "n_samples"]]
        raise ValueError(f"Unexpected sample count(s):\n{bad}")

    # Each model's confusion matrix must cover every evaluated grid-cell-day.
    count_cols = ["TP", "FP", "FN", "TN"]
    for _, row in master.iterrows():
        vals = pd.to_numeric(row[count_cols], errors="coerce")
        if vals.isna().any():
            raise ValueError(f"Missing contingency counts for {row['model']}")
        if int(vals.sum()) != N_EXPECTED:
            raise ValueError(
                f"Confusion-matrix counts for {row['model']} sum to {int(vals.sum())}, "
                f"not {N_EXPECTED}"
            )

    # KGE is undefined when predictions have zero variance; preserve as NaN/N/A.
    master.loc[master["model"].isin(["always_zero_rainfall", "always_zero"]), "KGE"] = pd.NA

    master = master.sort_values("CSI", ascending=False, na_position="last").reset_index(drop=True)
    master.to_csv(OUTPUT, index=False, float_format="%.8f")

    print("FINAL MASTER MODEL COMPARISON REBUILT")
    print(f"Rows: {len(master)} | samples per row: {N_EXPECTED}")
    print("Event threshold: 2.5 mm/day for every row")
    print("All contingency tables sum to 13,140: PASS")
    print("Duplicate model names: NONE")
    print("\n" + master.to_string(index=False))
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
