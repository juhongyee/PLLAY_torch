# src/utils/checkpoint.py
from __future__ import annotations
import torch
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

class CheckpointManager:
    """
    모델과 Optimizer 상태를 저장하고 불러오는 관리자.
    """
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = -float('inf') # 정확도 기준 (Loss 기준이면 float('inf'))

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        filename: str = "checkpoint.pt"
    ) -> None:
        """
        체크포인트 저장
        """
        # DataParallel 등으로 래핑된 경우 .module을 가져옴
        if hasattr(model, "module"):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

        state = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }

        # 1. 일반 체크포인트 저장 (덮어쓰기 or epoch별 저장)
        save_path = self.output_dir / filename
        torch.save(state, save_path)
        
        # 2. Best Model 별도 저장
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(state, best_path)
            logger.info(f"New best model saved to {best_path} (score={metrics.get('acc', 0):.4f})")

    def load(
        self,
        path: Union[str, Path],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device = torch.device("cpu")
    ) -> Dict[str, Any]:
        """
        체크포인트 로드
        Returns: loaded state dict (epoch, metrics 등 포함)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=device)

        # 모델 로드
        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        # 옵티마이저 로드 (학습 재개 시 필수)
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint