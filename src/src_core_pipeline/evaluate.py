"""Evaluation metrics: Continuous and categorical contingency verification for hydrometeorology."""

from typing import Dict, Tuple, Optional, List
import numpy as np
from scipy import stats

def compute_continuous_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes continuous hydrological regression metrics in mm.
    
    Args:
        y_true: Observed rainfall in mm (1D array)
        y_pred: Predicted rainfall in mm (1D array)
        
    Returns:
        Dict containing MAE, RMSE, PBias, NSE, KGE.
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    
    # Clip negative predictions to 0
    y_pred = np.maximum(y_pred, 0.0)

    # MAE and RMSE
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    # Percent Bias: 100 * sum(pred - obs) / sum(obs)
    sum_obs = np.sum(y_true)
    if sum_obs == 0:
        pbias = 0.0
    else:
        pbias = float(100.0 * np.sum(y_pred - y_true) / sum_obs)

    # Nash-Sutcliffe Efficiency (NSE)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        nse = 0.0 if ss_res == 0 else -999.0
    else:
        nse = float(1.0 - (ss_res / ss_tot))

    # Kling-Gupta Efficiency (KGE) (Gupta et al., 2009)
    # KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    if std_true == 0 or std_pred == 0:
        r = 0.0
        alpha = 0.0
    else:
        r_val, _ = stats.pearsonr(y_true, y_pred)
        r = float(r_val)
        alpha = float(std_pred / std_true)

    if mean_true == 0:
        beta = 1.0 if mean_pred == 0 else 2.0
    else:
        beta = float(mean_pred / mean_true)

    kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "PBias": pbias,
        "NSE": nse,
        "KGE": kge,
    }


def compute_contingency_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Computes categorical contingency verification scores at a given rainfall threshold.
    
    Args:
        y_true: Observed rainfall in mm
        y_pred: Predicted rainfall in mm
        threshold: Event threshold in mm (e.g. 2.5, 15.6, 64.5)
        
    Returns:
        Dict containing POD, FAR, CSI, HSS, hits, false_alarms, misses, correct_negatives.
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    y_pred = np.maximum(y_pred, 0.0)

    obs_event = y_true >= threshold
    pred_event = y_pred >= threshold

    a = int(np.sum(obs_event & pred_event))        # Hits
    b = int(np.sum((~obs_event) & pred_event))     # False alarms
    c = int(np.sum(obs_event & (~pred_event)))     # Misses
    d = int(np.sum((~obs_event) & (~pred_event)))   # Correct negatives
    total = a + b + c + d

    # Probability of Detection (POD) = a / (a + c)
    pod = float(a / (a + c)) if (a + c) > 0 else 0.0

    # False Alarm Ratio (FAR) = b / (a + b)
    far = float(b / (a + b)) if (a + b) > 0 else 0.0

    # Critical Success Index (CSI) = a / (a + b + c)
    csi = float(a / (a + b + c)) if (a + b + c) > 0 else 0.0

    # Heidke Skill Score (HSS)
    denom = float((a + c) * (c + d) + (a + b) * (b + d))
    if denom == 0:
        hss = 0.0
    else:
        hss = float(2.0 * (a * d - b * c) / denom)

    return {
        "threshold": threshold,
        "POD": pod,
        "FAR": far,
        "CSI": csi,
        "HSS": hss,
        "hits": a,
        "false_alarms": b,
        "misses": c,
        "correct_negatives": d,
        "event_count": a + c,
    }


def compute_wilcoxon_test(
    scores_baseline: np.ndarray,
    scores_proposed: np.ndarray,
    alternative: str = "less",
) -> Tuple[float, float]:
    """Paired Wilcoxon signed-rank test across 36 grid cells.
    
    Tests if scores_proposed is significantly different/better than scores_baseline.
    """
    diff = scores_proposed - scores_baseline
    if np.all(diff == 0):
        return 0.0, 1.0
    res = stats.wilcoxon(scores_proposed, scores_baseline, alternative=alternative)
    return float(res.statistic), float(res.pvalue)
