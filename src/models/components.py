# src/models/components.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import List

class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron block.
    [Linear -> ReLU -> Dropout] * (N-1) -> Linear
    """
    def __init__(
        self, 
        in_dim: int, 
        hidden_dims: List[int], 
        out_dim: int, 
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU()
    ):
        super().__init__()
        layers = []
        curr_dim = in_dim
        
        # Hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
            
        # Output layer
        layers.append(nn.Linear(curr_dim, out_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)