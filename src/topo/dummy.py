# src/topo/dummy.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Any

class DummyTopoExtractor(nn.Module):
    """
    Gudhi/Ripser의 출력을 흉내 내는 Dummy Extractor.
    실제 위상 계산은 하지 않지만, '그럴듯한' Persistence Diagram을 반환합니다.
    
    Output Schema:
      - "diagram": FloatTensor [B, max_points, 2] (Birth, Death)
      - "mask": BoolTensor [B, max_points] (True if point is valid)
    """
    def __init__(self, max_points: int = 64):
        super().__init__()
        self.max_points = max_points

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, C, H, W]
        """
        B = x.shape[0]
        device = x.device
        
        # 1. Hardcoded Base Diagrams (3개의 점이 있다고 가정)
        # 예: (Birth, Death) -> (0.1, 0.8), (0.2, 0.4), (0.5, 0.6)
        # 실제 Gudhi는 Birth < Death 규칙을 따릅니다.
        num_real_points = 3
        
        # [B, P, 2] 초기화 (0으로 패딩)
        diagram = torch.zeros(B, self.max_points, 2, device=device)
        mask = torch.zeros(B, self.max_points, dtype=torch.bool, device=device)
        
        # 2. 값 채워넣기 (배치마다 약간 다르게 노이즈 추가 - BatchNorm 에러 방지)
        # Point 1: Significant feature (Long lifetime)
        diagram[:, 0, 0] = 0.1 + torch.rand(B, device=device) * 0.05 # Birth
        diagram[:, 0, 1] = 0.8 + torch.rand(B, device=device) * 0.05 # Death
        
        # Point 2: Medium feature
        diagram[:, 1, 0] = 0.3 + torch.rand(B, device=device) * 0.05
        diagram[:, 1, 1] = 0.5 + torch.rand(B, device=device) * 0.05
        
        # Point 3: Noise (Short lifetime, close to diagonal)
        diagram[:, 2, 0] = 0.4 + torch.rand(B, device=device) * 0.05
        diagram[:, 2, 1] = 0.45 + torch.rand(B, device=device) * 0.05
        
        # 3. Mask 업데이트
        mask[:, :num_real_points] = True
        
        return {
            "diagram": diagram, # [B, 64, 2]
            "mask": mask        # [B, 64]
        }

class DummyTopoEmbedder(nn.Module):
    """
    Dummy Diagram을 받아서 Embedding으로 변환하는 흉내를 냄.
    """
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, topo_data: Dict[str, Any], batch_size: int, device: torch.device) -> torch.Tensor:
        # 들어온 데이터를 확인은 하지만, 계산은 하지 않고 랜덤 반환
        # (실제 구현에선 topo_data["diagram"]을 지지고 볶아서 벡터로 만듦)
        
        # Check Contract
        if "diagram" in topo_data:
            diag = topo_data["diagram"]
            # 간단한 assert (Shape 확인용)
            assert diag.ndim == 3 and diag.shape[2] == 2, f"Bad diagram shape: {diag.shape}"
            
        return torch.randn(batch_size, self.out_dim, device=device)