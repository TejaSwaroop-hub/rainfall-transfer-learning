"""Loss functions for Rainfall Transfer Learning."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassWeightedHuberLoss(nn.Module):
    """Class-weighted Huber loss on log-transformed rainfall y' = ln(1 + y).
    
    Weights are assigned dynamically based on the true rainfall value in mm:
        - No rain (< 0.1 mm): 1.0
        - Very light (0.1 - 2.5 mm): 1.0
        - Light (2.5 - 15.6 mm): 1.5
        - Moderate (15.6 - 64.5 mm): 2.5
        - Heavy (64.5 - 115.6 mm): 4.0
        - Very heavy (>= 115.6 mm): 4.0
    
    For Ablation A3, weighted=False uses standard unweighted Huber loss.
    """
    def __init__(self, delta: float = 1.0, weighted: bool = True):
        super().__init__()
        self.delta = delta
        self.weighted = weighted

    def get_sample_weights(self, y_true_transformed: torch.Tensor) -> torch.Tensor:
        """Computes sample weights based on the original rainfall scale in mm.
        
        Args:
            y_true_transformed: Tensor of shape (B,) or (B, 1) in y' = ln(1 + y).
            
        Returns:
            Tensor of same shape with weights w_c.
        """
        if not self.weighted:
            return torch.ones_like(y_true_transformed)
            
        # Invert transform: y = exp(y') - 1
        y_true_mm = torch.clamp(torch.exp(y_true_transformed) - 1.0, min=0.0)
        
        weights = torch.ones_like(y_true_mm)
        # 0.0 to 2.5 mm is weight 1.0 (default)
        weights = torch.where((y_true_mm >= 2.5) & (y_true_mm < 15.6), torch.tensor(1.5, device=weights.device), weights)
        weights = torch.where((y_true_mm >= 15.6) & (y_true_mm < 64.5), torch.tensor(2.5, device=weights.device), weights)
        weights = torch.where(y_true_mm >= 64.5, torch.tensor(4.0, device=weights.device), weights)
        
        return weights

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculates class-weighted Huber loss.
        
        Args:
            y_pred: Predicted transformed rain (B, 1) or (B,)
            y_true: Ground-truth transformed rain (B, 1) or (B,)
            
        Returns:
            Scalar loss tensor.
        """
        weights = self.get_sample_weights(y_true)
        # Element-wise Huber loss
        elementwise_huber = F.huber_loss(y_pred, y_true, reduction='none', delta=self.delta)
        weighted_loss = weights * elementwise_huber
        return weighted_loss.mean()
