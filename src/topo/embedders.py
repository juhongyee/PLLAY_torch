from __future__ import annotations
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gudhi.representations import Landscape

# ==========================================
# 1. Structure Learning Layer (g_theta)
# ==========================================

class LandscapeWeightedAverage(nn.Module):
    """
    [Algorithm 1 Step 2]
    Weighted average of landscape layers:
    lambda_bar(t) = sum( omega_k * lambda_k(t) )
    
    Constraint: omega_k > 0, sum(omega_k) = 1
    """
    def __init__(self, num_layers: int):
        super().__init__()
        self.omega_logits = nn.Parameter(torch.ones(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, num_layers, num_bins]
        # sum = 1 제약을 위해 softmax 사용
        weights = F.softmax(self.omega_logits, dim=0) # [num_layers]
        
        # Weighted Sum 계산
        # weights를 [1, num_layers, 1]로 reshape하여 브로드캐스팅
        weights_expanded = weights.view(1, -1, 1)
        
        # Sum over layers (dim=1)
        # out: [B, num_bins]
        out = torch.sum(x * weights_expanded, dim=1)
        
        return out

class AffineStructureLearner(nn.Module):
    """
    g_theta(x) = w * x + b
    단순한 선형 변환을 Element-wise로 학습합니다.
    """
    def __init__(self, input_shape: Tuple[int, int]):
        super().__init__()
        # input_shape: (num_layers, num_bins)
        self.weight = nn.Parameter(torch.ones(input_shape))
        self.bias = nn.Parameter(torch.zeros(input_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, num_layers, num_bins]
        return x * self.weight + self.bias


class LogStructureLearner(nn.Module):
    """
    g_theta(x) = log( |w * x + b| + epsilon )
    PLLAY 논문에서 제안한 Logarithmic Transformation.
    Landscape의 스케일 차이를 완화하고, 중요한 특징을 강조합니다.
    """
    def __init__(self, input_shape: Tuple[int, int], epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        # 초기화: w=1, b=0으로 시작
        self.weight = nn.Parameter(torch.ones(input_shape))
        self.bias = nn.Parameter(torch.zeros(input_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, num_layers, num_bins]
        # |wx + b| 계산
        linear = x * self.weight + self.bias
        # Log transform
        return torch.log(torch.abs(linear) + self.epsilon)


# ==========================================
# 2. Main Embedder
# ==========================================

class TopoEmbedder(nn.Module):
    """
    Role: Diagram -> Landscape -> Structure Learning -> Feature Vector
    """
    def __init__(
        self,
        num_layers: int = 4,
        num_bins: int = 100,
        sample_range: Tuple[float, float] = (0.0, 1.0), # [Important] Landscape X축 범위 고정
        structure_type: str = "affine", # 'log' or 'affine' or 'none'
        out_dim: int = 64,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_bins = num_bins
        self.sample_range = sample_range
        
        # 1. GUDHI Landscape Transformer (CPU based)
        # sample_range를 고정해야 배치 간, Train/Test 간 일관성이 유지됩니다.
        self.landscape_transformer = Landscape(
            num_landscapes=num_layers,
            resolution=num_bins,
            sample_range=sample_range
        )
        
        # 2. Structure Learner (g_theta)
        input_shape = (num_layers, num_bins)
        if structure_type == "log":
            self.structure_learner = LogStructureLearner(input_shape)
        elif structure_type == "affine":
            self.structure_learner = AffineStructureLearner(input_shape)
        else:
            self.structure_learner = nn.Identity()
            
        # 3. Feature Aggregation (Vectorization)
        # Landscape [L, Bins] -> Flatten -> Linear -> Output
        flat_dim = num_layers * num_bins
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, out_dim),
            nn.BatchNorm1d(out_dim), # [Important] Scale normalization
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def _generate_landscapes(self, diagrams: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Tensor Diagram [B, N, 2] -> GUDHI -> Tensor Landscape [B, L, Bins]
        NOTE: 이 과정은 CPU 연산을 포함하므로 병목이 될 수 있습니다.
        """
        B = diagrams.shape[0]
        device = diagrams.device
        
        # CPU로 이동 (GUDHI는 GPU 미지원)
        diags_np = diagrams.detach().cpu().numpy()
        masks_np = masks.detach().cpu().numpy().astype(bool)
        
        clean_diagrams = []
        for i in range(B):
            # Mask를 이용해 Valid Point만 추출
            valid_pts = diags_np[i][masks_np[i]]
            
            # GUDHI Landscape는 빈 리스트가 들어오면 에러가 날 수 있음.
            # 방어 코드: 점이 없으면 더미 점(range 밖)이나 0을 넣어야 함.
            if len(valid_pts) == 0:
                # (0,0) 점 하나 추가 (Landscape 계산 시 0이 됨)
                clean_diagrams.append(np.array([[0.0, 0.0]]))
            else:
                clean_diagrams.append(valid_pts)
        
        # GUDHI Transformation
        # Output: [B, num_layers * num_bins] (Flattened)
        landscapes_flat = self.landscape_transformer.fit_transform(clean_diagrams)
        
        # Reshape: [B, Layers, Bins]
        landscapes = landscapes_flat.reshape(B, self.num_layers, self.num_bins)
        
        # Tensor 변환 & Device 복귀
        return torch.from_numpy(landscapes).float().to(device)

    def forward(self, topo_data: dict) -> torch.Tensor:
            diagram = topo_data["diagram"]
            mask = topo_data["mask"]
            
            # 1. Generate Grid Landscapes [B, L, Bins]
            x = self._generate_landscapes(diagram, mask) 
            
            # 2. Weighted Average [B, L, Bins] -> [B, Bins]
            x_avg = self.weighted_avg(x)
            
            # 3. Structure Learning [B, Bins]
            x_mapped = self.structure_learner(x_avg)
            
            # 4. Final Projection
            out = self.head(x_mapped)
            
            return out