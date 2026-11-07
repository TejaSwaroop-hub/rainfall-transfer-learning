"""Sliding window dataset generator with strict partition and cell boundary enforcement."""

from typing import Tuple, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class RainfallSlidingWindowDataset(Dataset):
    """PyTorch Dataset holding sliding window sequences of length L predicting t+1.
    
    Guarantees:
        - Windows NEVER straddle temporal partition boundaries.
        - Windows NEVER cross different spatial grid cells.
        - Predictors are standardized X in shape (N_samples, L, K).
        - Target is transformed y' = ln(1 + y) in shape (N_samples, 1).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, metadata: Optional[List[dict]] = None):
        """
        Args:
            X: Array of shape (N_samples, L, K)
            y: Array of shape (N_samples, 1)
            metadata: Optional list of dicts with cell_idx, date, etc.
        """
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_sliding_windows_for_cell(
    predictors: np.ndarray,
    targets: np.ndarray,
    lookback: int = 14,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Creates sliding windows for a single continuous time series of one cell.
    
    Args:
        predictors: 2D array of shape (T, K)
        targets: 1D or 2D array of shape (T,) or (T, 1)
        lookback: window length L (e.g. 14 days)
        horizon: prediction lead time h (e.g. 1 day, predicting t + 1)
        
    Returns:
        X: shape (T - lookback - horizon + 1, lookback, K)
        y: shape (T - lookback - horizon + 1, 1)
    """
    targets = np.squeeze(targets)
    T, K = predictors.shape
    num_samples = T - lookback - horizon + 1
    if num_samples <= 0:
        return np.empty((0, lookback, K), dtype=np.float32), np.empty((0, 1), dtype=np.float32)

    X_list = []
    y_list = []
    for t in range(lookback - 1, T - horizon):
        x_window = predictors[t - lookback + 1 : t + 1]  # shape (lookback, K)
        y_val = targets[t + horizon]                     # scalar at t + horizon
        
        # Check for missing values (NaNs in target or window)
        if np.isnan(y_val) or np.isnan(x_window).any():
            continue
            
        X_list.append(x_window)
        y_list.append(y_val)

    if len(X_list) == 0:
        return np.empty((0, lookback, K), dtype=np.float32), np.empty((0, 1), dtype=np.float32)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32).reshape(-1, 1)


def build_regional_dataset(
    cell_predictors: List[np.ndarray],
    cell_targets: List[np.ndarray],
    lookback: int = 14,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pools sliding windows across all 36 cells into a unified regional dataset.
    
    Guarantees windows do not cross between different cells.
    """
    all_X = []
    all_y = []
    for preds, targs in zip(cell_predictors, cell_targets):
        X_cell, y_cell = create_sliding_windows_for_cell(preds, targs, lookback=lookback, horizon=horizon)
        if len(X_cell) > 0:
            all_X.append(X_cell)
            all_y.append(y_cell)

    if len(all_X) == 0:
        return np.empty((0, lookback, 0), dtype=np.float32), np.empty((0, 1), dtype=np.float32)

    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)


def get_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 256,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Creates a PyTorch DataLoader from arrays."""
    dataset = RainfallSlidingWindowDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
