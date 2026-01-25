# src/data/datasets.py
from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import torch
from torchvision.datasets import MNIST
from torchvision import transforms as T

from src.data.transforms import CorruptAndNoise
from src.config import DataConfig

class PllayMNIST(MNIST):
    def __init__(self, cfg: DataConfig, split: str = "train", download: bool = True):
        root = cfg.root
        is_train = (split == "train")
        
        # 1. Transform 설정 (노이즈 등)
        tf_list = [T.ToTensor()]
        
        # Config에 따라 노이즈 적용 (Train/Test 모두 제어 가능)
        if (cfg.augmentation.corrupt_prob > 0 or cfg.augmentation.noise_prob > 0):
            tf_list.append(CorruptAndNoise(
                corrupt_prob=cfg.augmentation.corrupt_prob,
                noise_prob=cfg.augmentation.noise_prob,
                background_threshold=0.01,
                noise_high=1.0
            ))
        
        if cfg.augmentation.normalize:
            tf_list.append(T.Normalize((0.1307,), (0.3081,)))

        transform = T.Compose(tf_list)

        super().__init__(root=root, train=is_train, transform=transform, download=download)

        # 2. [Hybrid] Topo Feature 로드 시도
        self.topo_data = None
        self.use_offline_topo = False
        
        topo_path = None
        if is_train and cfg.train_topo_path:
            topo_path = Path(cfg.train_topo_path)
        elif not is_train and cfg.test_topo_path:
            topo_path = Path(cfg.test_topo_path)
            
        # 파일이 있으면 로드 (Track A), 없으면 패스 (Track B)
        if topo_path and topo_path.exists():
            print(f"[{split}] Loading offline topo features from {topo_path}...")
            try:
                self.topo_data = torch.load(topo_path)
                # 데이터 개수 체크
                if len(self.topo_data) != len(self.data):
                    print(f"Warning: Topo count ({len(self.topo_data)}) != Image count ({len(self.data)})")
                else:
                    self.use_offline_topo = True
            except Exception as e:
                print(f"Error loading topo file: {e}. Switching to Online Mode.")
        else:
            if topo_path:
                print(f"Warning: Topo file {topo_path} not found. Switching to Online Mode.")
            else:
                # 경로가 아예 설정 안 된 경우 -> 자연스럽게 Online Mode
                pass

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Parent class returns (image_tensor, label)
        img, target = super().__getitem__(index)
        
        item = {
            "image": img,
            "target": target,
            "id": index,
        }
        
        # Track A: 전처리 데이터가 있으면 넣어줌
        if self.use_offline_topo and self.topo_data is not None:
            topo_feat = self.topo_data[index]
            item["topo"] = {
                "topo_feat": topo_feat
            }
            
        # Track B: 없으면 "topo" 키 자체를 안 넣음 -> Model이 알아서 계산
            
        return item