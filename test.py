# src/test.py
from __future__ import annotations
import sys
import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

# 프로젝트 루트 경로 추가 (모듈 import를 위해)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import load_config
from src.utils import get_device, set_seed
from src.data import build_dataloader
from src.models import TopoMLPClassifier

def test(cfg_path: str, ckpt_path: str):
    # 1. Config & Setup
    cfg = load_config(cfg_path)
    set_seed(cfg.seed)
    dev_info = get_device(cfg.device)
    device = dev_info.device
    
    print(f">>> Loading Configuration from {cfg_path}")
    print(f">>> Loading Checkpoint from {ckpt_path}")
    
    # 2. Data Loader (Test Split)
    # shuffle=False는 평가 시 필수
    test_loader = build_dataloader(cfg.data, split="test", shuffle=False)
    
    # 3. Model Build
    # 학습 때와 동일한 모델 구조 생성
    model = TopoMLPClassifier(cfg.model)
    model.to(device)
    
    # 4. Load Weights
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 체크포인트 저장 방식에 따라 state_dict 추출
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint # 전체가 state_dict인 경우
        
    model.load_state_dict(state_dict)
    print(">>> Model weights loaded successfully.")
    
    # 5. Evaluation Loop
    model.eval()
    correct = 0
    total = 0
    
    print(">>> Start Testing...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Move data to device
            # (Trainer의 _to_device 로직을 간단히 구현)
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward
            outputs = model(batch)
            
            # Accuracy Calculation
            # Trainer와 동일하게 outputs["logits"] 사용
            logits = outputs["logits"]
            targets = batch["target"]
            
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
    accuracy = correct / total * 100
    print(f"\n==============================")
    print(f"Final Test Accuracy: {accuracy:.2f}%")
    print(f"==============================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PLLAY Test Script")
    parser.add_argument("config", type=str, help="Path to config file (e.g., configs/default.yaml)")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file (e.g., outputs/.../best_model.pt)")
    
    args = parser.parse_args()
    
    test(args.config, args.checkpoint)