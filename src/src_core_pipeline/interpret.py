"""Interpretability Analysis: Layer-wise Relative Weight Shift and Permutation Feature Importance."""

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
from src.windows import build_regional_dataset, RainfallSlidingWindowDataset
from src.evaluate import compute_continuous_metrics

def compute_layer_weight_shift(
    pretrain_weights_path: str,
    finetuned_weights_path: str,
) -> dict:
    """Computes relative Frobenius norm shift per layer:
    delta_l = ||theta_ft - theta_pre||_2 / ||theta_pre||_2
    """
    pre_state = torch.load(pretrain_weights_path, map_location="cpu")
    ft_state = torch.load(finetuned_weights_path, map_location="cpu")

    def layer_norm_diff(param_names):
        diff_sq = 0.0
        norm_sq = 0.0
        for name in param_names:
            if name in pre_state and name in ft_state:
                p_pre = pre_state[name].float()
                p_ft = ft_state[name].float()
                diff_sq += torch.sum((p_ft - p_pre) ** 2).item()
                norm_sq += torch.sum(p_pre ** 2).item()
        return float(np.sqrt(diff_sq) / (np.sqrt(norm_sq) + 1e-8))

    l1_names = [n for n in pre_state.keys() if "_l0" in n]
    l2_names = [n for n in pre_state.keys() if "_l1" in n]
    head_names = [n for n in pre_state.keys() if "fc." in n]

    delta_1 = layer_norm_diff(l1_names)
    delta_2 = layer_norm_diff(l2_names)
    delta_head = layer_norm_diff(head_names)

    print("\nLayer-wise Relative Weight Shift (Frobenius norm):")
    print(f"  GRU Layer 1:      delta_1    = {delta_1:.4f} ({delta_1*100:.1f}%)")
    print(f"  GRU Layer 2:      delta_2    = {delta_2:.4f} ({delta_2*100:.1f}%)")
    print(f"  Dense Output Head: delta_head = {delta_head:.4f} ({delta_head*100:.1f}%)")

    return {
        "delta_layer1": delta_1,
        "delta_layer2": delta_2,
        "delta_head": delta_head,
    }


def compute_permutation_importance(
    model: RainfallRecurrentModel,
    X_test: np.ndarray,
    y_test_transformed: np.ndarray,
    feature_names: list,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> list:
    """Computes Permutation Feature Importance on held-out test partition."""
    model.eval()

    def eval_rmse(X_eval):
        loader = DataLoader(RainfallSlidingWindowDataset(X_eval, y_test_transformed), batch_size=batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for X_b, _ in loader:
                p = model(X_b.to(device))
                preds.append(p.cpu().numpy())
        preds_mm = np.maximum(np.expm1(np.concatenate(preds, axis=0)), 0.0)
        true_mm = np.maximum(np.expm1(y_test_transformed), 0.0)
        return compute_continuous_metrics(true_mm, preds_mm)["RMSE"]

    base_rmse = eval_rmse(X_test)
    print(f"\nPermutation Importance: Base Test RMSE = {base_rmse:.4f} mm")

    results = []
    np.random.seed(42)
    K = X_test.shape[-1]

    for k in range(K):
        feat_name = feature_names[k]
        X_perm = X_test.copy()
        
        # Shuffle across samples for this feature channel across all lookback steps
        perm_idx = np.random.permutation(len(X_perm))
        X_perm[:, :, k] = X_perm[perm_idx, :, k]

        perm_rmse = eval_rmse(X_perm)
        rel_increase = float((perm_rmse - base_rmse) / (base_rmse + 1e-8))
        
        results.append({
            "feature": feat_name,
            "base_rmse": base_rmse,
            "perm_rmse": perm_rmse,
            "rel_increase": rel_increase,
        })
        print(f"  Feature {feat_name:12s}: Permuted RMSE = {perm_rmse:.4f} mm (+{rel_increase*100:.2f}%)")

    # Sort descending by relative increase
    results.sort(key=lambda x: x["rel_increase"], reverse=True)
    return results
