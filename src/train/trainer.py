# src/train/trainer.py
from __future__ import annotations
import sys
from typing import Dict, Any
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

try:
    from src.config import AppConfig
except ImportError:
    AppConfig = Any

from src.utils.checkpoint import CheckpointManager

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        cfg: AppConfig,
        logger: Any,
        ckpt_manager: CheckpointManager,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.cfg = cfg
        self.logger = logger
        self.ckpt_manager = ckpt_manager
        
        use_amp = getattr(cfg.train, 'amp', False)
        self.scaler = GradScaler(enabled=use_amp)

    def _to_device(self, batch: Any) -> Any:
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self._to_device(v) for v in batch]
        return batch

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        use_amp = getattr(self.cfg.train, 'amp', False)
        grad_clip = getattr(self.cfg.train, 'grad_clip_norm', 0.0)

        pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", file=sys.stdout, dynamic_ncols=True)
        
        for step, batch in enumerate(pbar):
            batch = self._to_device(batch)
            targets = batch["target"]
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=use_amp):
                outputs = self.model(batch)
                loss = self.loss_fn(outputs, targets)

            self.scaler.scale(loss).backward()
            
            if grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    grad_clip
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            bs = targets.size(0)
            total_loss += loss.item() * bs
            
            # 정확도 계산을 위해 여기서만 잠깐 꺼내 봄
            logits = outputs["logits"]
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
            
            total_samples += bs
            
            if step % 10 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples
        
        return {"loss": avg_loss, "acc": accuracy}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, epoch: int, split: str = "val") -> Dict[str, float]:
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        use_amp = getattr(self.cfg.train, 'amp', False)
        
        for batch in loader:
            batch = self._to_device(batch)
            targets = batch["target"]
            
            with autocast(enabled=use_amp):
                outputs = self.model(batch)
                loss = self.loss_fn(outputs, targets)
            
            bs = targets.size(0)
            total_loss += loss.item() * bs
            
            logits = outputs["logits"]
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total_samples += bs

        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples
        
        return {"loss": avg_loss, "acc": accuracy}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, start_epoch: int = 1):
        self.logger.info(">>> Start Training")
        best_acc = 0.0
        
        total_epochs = getattr(self.cfg.train, 'epochs', 10)
        eval_every = getattr(self.cfg.train, 'eval_every', 1)
        save_every = getattr(self.cfg.train, 'save_every', 1)

        for epoch in range(start_epoch, total_epochs + 1):
            train_metrics = self.train_one_epoch(train_loader, epoch)
            
            msg = f"[Epoch {epoch}] Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['acc']:.4f}"
            self.logger.info(msg)
            
            val_metrics = {}
            is_best = False
            
            if epoch % eval_every == 0:
                val_metrics = self.evaluate(val_loader, epoch, split="val")
                
                msg = f"[Epoch {epoch}] Val  : loss={val_metrics['loss']:.4f}, acc={val_metrics['acc']:.4f}"
                self.logger.info(msg)
                
                if val_metrics['acc'] > best_acc:
                    best_acc = val_metrics['acc']
                    is_best = True
                    self.logger.info(f"New Best Accuracy: {best_acc:.4f} found!")

            if epoch % save_every == 0 or is_best:
                self.ckpt_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics={**train_metrics, **val_metrics},
                    is_best=is_best
                )