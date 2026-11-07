"""Baseline Models Implementation.
Implements:
  - M0: Persistence (rain at t as forecast for t+1)
  - M1: Day-of-year Climatology (mean rainfall for that DOY across IMD pool)
  - M2: From-Scratch GRU (trained only on N years of IMD observations)
  - M3: From-Scratch LSTM (trained only on N years of IMD observations)
  - M4: Zero-Shot ERA5 GRU (evaluates theta_pre directly on IMD)
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models import RainfallRecurrentModel
from src.losses import ClassWeightedHuberLoss
from src.windows import build_regional_dataset, RainfallSlidingWindowDataset
from src.evaluate import compute_continuous_metrics

def evaluate_persistence(test_y_raw: list, lookback: int = 14) -> np.ndarray:
    """M0: Forecasts rainfall at t+1 as rainfall at t."""
    preds = []
    for cell_raw in test_y_raw:
        # windowing starts at lookback - 1, predicts t+1. Last observed day in window is day t.
        T = len(cell_raw)
        for t in range(lookback - 1, T - 1):
            preds.append(cell_raw[t])
    return np.array(preds, dtype=np.float32)

def evaluate_climatology(pool_y_raw: list, dates_pool: list, dates_test: list, lookback: int = 14) -> np.ndarray:
    """M1: Day-of-year Climatology computed across the IMD training pool."""
    pool_doy = np.array([int(d.split("-")[1]) * 100 + int(d.split("-")[2]) for d in dates_pool])
    test_doy = np.array([int(d.split("-")[1]) * 100 + int(d.split("-")[2]) for d in dates_test])

    # Flatten all cells to compute mean rain per day-of-year
    doy_means = {}
    pool_all = np.concatenate(pool_y_raw)
    doy_rep = np.tile(pool_doy, len(pool_y_raw))

    for u_doy in np.unique(test_doy):
        mask = doy_rep == u_doy
        doy_means[u_doy] = float(np.mean(pool_all[mask])) if mask.sum() > 0 else 0.0

    preds = []
    n_days_test = len(dates_test)
    for _ in range(len(pool_y_raw)):  # 36 cells
        for t in range(lookback - 1, n_days_test - 1):
            target_doy = test_doy[t + 1]
            preds.append(doy_means.get(target_doy, 0.0))

    return np.array(preds, dtype=np.float32)

def train_scratch_model(
    cell_type: str = "GRU",
    n_years: int = 5,
    seed: int = 0,
    config_path: str = "configs/default.yaml",
    fast: bool = False,
) -> str:
    """Trains recurrent model (GRU or LSTM) from scratch on N years of IMD target data."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not fast else "cpu")
    print(f"\n--- Training Scratch {cell_type} | N={n_years}yr | Seed={seed} ---")

    processed_dir = cfg["paths"]["processed"]
    finetune_data = np.load(os.path.join(processed_dir, "finetune_data.npz"), allow_pickle=True)
    imd_pool_X = finetune_data["imd_pool_X"]
    imd_pool_y = finetune_data["imd_pool_y"]
    imd_val_X = finetune_data["imd_val_X"]
    imd_val_y = finetune_data["imd_val_y"]

    with open(os.path.join(processed_dir, "selected_features.json"), "r") as f:
        feat_info = json.load(f)
    data_pt = np.load(os.path.join(processed_dir, "pretrain_data.npz"), allow_pickle=True)
    all_names = list(data_pt["feature_names"])
    selected_idx = [all_names.index(name) for name in feat_info["selected_features"]]

    pool_X_filt = [c[:, selected_idx] for c in imd_pool_X]
    val_X_filt = [c[:, selected_idx] for c in imd_val_X]

    days_to_keep = int(round(n_years * 365.25))
    if n_years == 16:
        days_to_keep = len(pool_X_filt[0])

    pool_X_sub = [c[-days_to_keep:] for c in pool_X_filt]
    pool_y_sub = [c[-days_to_keep:] for c in imd_pool_y]

    lookback = cfg["model"]["lookback_window"]
    X_train, y_train = build_regional_dataset(pool_X_sub, pool_y_sub, lookback=lookback, horizon=1)
    X_val, y_val = build_regional_dataset(val_X_filt, imd_val_y, lookback=lookback, horizon=1)

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(RainfallSlidingWindowDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(RainfallSlidingWindowDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = RainfallRecurrentModel(
        input_dim=len(selected_idx),
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        cell_type=cell_type,
    ).to(device)

    lr = cfg["training"]["finetune"]["lr"]
    max_epochs = 3 if fast else cfg["training"]["finetune"]["max_epochs"]
    patience = 2 if fast else cfg["training"]["finetune"]["patience"]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = ClassWeightedHuberLoss(delta=cfg["model"]["huber_delta"], weighted=True)

    models_dir = os.path.join(cfg["paths"]["results"], "models")
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, f"theta_scratch_{cell_type.lower()}_N{n_years}_seed{seed}.pt")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                val_loss += criterion(preds, y_batch).item() * len(X_batch)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {total_loss/len(train_loader.dataset):.5f} | Val Loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    print(f"Scratch {cell_type} training finished. Saved to {save_path}")
    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    # Smoke test scratch models
    train_scratch_model(cell_type="GRU", n_years=5, fast=args.fast)
    train_scratch_model(cell_type="LSTM", n_years=5, fast=args.fast)
