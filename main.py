# src/main.py
from __future__ import annotations
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.optim as optim

from src.config import load_config, dump_resolved_config
from src.utils import set_seed, get_device, setup_logger, CheckpointManager

from src.data import build_dataloader
from src.models import TopoMLPClassifier
from src.losses import ClassificationLoss
from src.train import Trainer

def main(cfg_path: str) -> None:
    # 1. Config & Logger
    cfg = load_config(cfg_path)
    logger = setup_logger(name="pllay_torch", log_dir=cfg.output_dir, console=True, file=True)
    
    # 2. Infra (Seed, Device, Checkpoint)
    set_seed(cfg.seed)
    dev_info = get_device(cfg.device)
    ckpt_manager = CheckpointManager(output_dir=cfg.output_dir)
    
    logger.info(f"Initialized Experiment: {cfg.run_name}")
    logger.info(f"Device: {dev_info.device}")

    # Config 백업
    if cfg.output_dir:
        dump_resolved_config(cfg, f"{cfg.output_dir}/resolved_config.yaml")

    # 3. Data Loaders
    logger.info("Building dataloaders...")
    train_loader = build_dataloader(cfg.data, split="train", shuffle=True)
    val_loader = build_dataloader(cfg.data, split="test", shuffle=False) # MNIST는 Val이 따로 없어 Test 사용

    # 4. Model Build
    logger.info("Building model...")
    model = TopoMLPClassifier(cfg.model)
    model.to(dev_info.device)
    
    # 5. Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=cfg.train.lr, 
        weight_decay=cfg.train.weight_decay
    )
    loss_fn = ClassificationLoss()

    # 6. Trainer Init
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=dev_info.device,
        cfg=cfg,
        logger=logger,
        ckpt_manager=ckpt_manager
    )

    # 7. Start Training
    logger.info(">>> Start Training Loop")
    trainer.fit(train_loader, val_loader)
    logger.info("<<< Training Finished")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/main.py configs/default.yaml")
        sys.exit(1)
    main(sys.argv[1])