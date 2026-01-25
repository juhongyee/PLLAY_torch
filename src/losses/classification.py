# src/losses/classification.py
from __future__ import annotations
from typing import Dict, Union
import torch
import torch.nn as nn

class ClassificationLoss(nn.Module):
    """
    Standard CrossEntropyLoss.
    Wrapper designed to handle Model Output Dictionary.
    """
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self, 
        outputs: Dict[str, torch.Tensor], 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            outputs: Model output dict (must contain "logits")
            targets: Label Tensor (from Trainer)
        """
        logits = outputs["logits"]

        # Loss 계산
        total_loss = self.ce_loss(logits, targets)

        return total_loss