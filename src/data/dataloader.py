# src/data/dataloader.py
from __future__ import annotations
from typing import Dict, Any, List
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from src.config import DataConfig
from src.data.datasets import PllayMNIST

# TODO
def dict_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function ensures dictionaries are stacked correctly.
    Basic PyTorch default_collate handles list of dicts -> dict of stacked tensors well,
    but we keep this explicitly for future extension (e.g., handling variable size diagrams).
    """
    return default_collate(batch)

def build_dataloader(
    cfg: DataConfig,
    split: str = "train",
    shuffle: bool = True
) -> DataLoader:
    
    # 1. Dataset 생성
    dataset = PllayMNIST(cfg, split=split, download=True)
    
    # 2. DataLoader 생성
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=dict_collate_fn,
        drop_last=(split == "train") # 학습 때는 batch norm 안정성을 위해 drop_last 추천
    )
    
    return loader