"""Neural network architectures for Rainfall Transfer Learning: GRU and LSTM backbones."""

from typing import Literal, Tuple
import torch
import torch.nn as nn

class RainfallRecurrentModel(nn.Module):
    """Stacked recurrent neural network (GRU or LSTM) for rainfall prediction.
    
    Architecture:
        - 2 recurrent layers (default: 64 hidden units each)
        - Dropout between and after recurrent layers (default: 0.2)
        - Fully connected linear output layer predicting y'(t+1) = ln(1 + y)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        cell_type: Literal["GRU", "LSTM"] = "GRU",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type.upper()
        self.dropout_rate = dropout

        # Recurrent backbone
        rnn_dropout = dropout if num_layers > 1 else 0.0
        if self.cell_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        elif self.cell_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        else:
            raise ValueError(f"Unsupported cell_type: {cell_type}. Choose 'GRU' or 'LSTM'.")

        self.post_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output prediction tensor of shape (batch_size, 1)
        """
        # rnn output: (batch_size, seq_len, hidden_dim)
        if self.cell_type == "GRU":
            out, _ = self.rnn(x)
        else:
            out, _ = self.rnn(x)
        
        # Take the hidden representation of the final time step
        last_step = out[:, -1, :]  # (batch_size, hidden_dim)
        last_step = self.post_dropout(last_step)
        y_pred = self.fc(last_step)  # (batch_size, 1)
        return y_pred

    def set_fine_tuning_strategy(self, strategy: Literal["S1", "S2", "S3"]):
        """Configures parameter freezing for Stage 2 fine-tuning.
        
        S1: Fine-tune all layers (all requires_grad = True)
        S2: Freeze layer 1 of the recurrent backbone
        S3: Freeze all recurrent layers (train only dense head)
        """
        # First enable all
        for param in self.parameters():
            param.requires_grad = True

        if strategy == "S1":
            pass  # All trainable
        elif strategy == "S2":
            # Freeze layer 0 parameters of the RNN
            for name, param in self.rnn.named_parameters():
                if "_l0" in name:
                    param.requires_grad = False
        elif strategy == "S3":
            # Freeze all RNN parameters
            for param in self.rnn.parameters():
                param.requires_grad = False
        else:
            raise ValueError(f"Unknown fine-tuning strategy: {strategy}. Choose S1, S2, or S3.")

    def enable_mc_dropout(self):
        """Enables dropout layers during inference for Monte-Carlo Dropout."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def count_parameters(self) -> int:
        """Returns the total number of trainable and non-trainable parameters."""
        return sum(p.numel() for p in self.parameters())
