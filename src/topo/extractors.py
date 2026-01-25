from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import gudhi

class ExactWeightedDTM(nn.Module):
    def __init__(self, m0: float = 0.01, r: float = 2.0):
        super().__init__()
        self.m0 = m0
        self.r = r

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W] (Normalized 0~1 recommended)
        Returns:
            dtm: [B, H, W]
        """
        B, C, H, W = images.shape
        device = images.device
        N = H * W
        
        # 1. Flatten Images to [B, N] (Pixel Weights)
        weights = images.view(B, N)
        
        # 2. Coordinate Grid Construction [N, 2]
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device), 
            torch.arange(W, device=device), 
            indexing='ij'
        )
        coords = torch.stack([y_coords.flatten(), x_coords.flatten()], dim=1).float()
        
        # 3. Pairwise Distance Matrix Calculation [N, N]
        # This is the heavy part: O(N^2)
        dists = torch.cdist(coords, coords, p=2) 
        dists_pow = dists.pow(self.r) 

        # 4. Sort Neighbors by Distance
        # For every pixel, sort all other pixels by distance
        sorted_dists_pow, sorted_indices = torch.sort(dists_pow, dim=1)
        
        dtm_list = []
        
        for b in range(B):
            w = weights[b] # [N]
            
            # Total mass check
            total_mass = w.sum()
            if total_mass == 0:
                dtm_list.append(torch.zeros(H, W, device=device))
                continue
                
            m_target = self.m0 * total_mass
            
            # Gather weights of sorted neighbors
            w_sorted = w[sorted_indices] # [N, N]
            
            # 5. Weighted DTM Calculation logic
            w_cumsum = torch.cumsum(w_sorted, dim=1)
            prev_cumsum = w_cumsum - w_sorted
            remains = m_target - prev_cumsum
            
            # Interpolation weight (k_w in paper)
            w_effective = torch.clamp(torch.min(w_sorted, remains), min=0.0)
            
            weighted_dist_sum = torch.sum(w_effective * sorted_dists_pow, dim=1)
            
            # Final formula
            dtm_val = (weighted_dist_sum / m_target).pow(1.0 / self.r)
            dtm_list.append(dtm_val.view(H, W))
            
        return torch.stack(dtm_list)


class GudhiCubicalExtractor(nn.Module):
    """
    Role: Image -> Exact Weighted DTM -> Cubical Persistence -> Diagram Tensor
    """
    def __init__(
        self, 
        max_points: int = 64,
        m0: float = 0.01,
        r: float = 2.0,
        homology_dims: list = [0, 1], # [Check] 리스트 지원
        replace_inf_value: float = 99.0 
    ):
        super().__init__()
        self.max_points = max_points
        self.homology_dims = homology_dims
        self.replace_inf_value = replace_inf_value
        
        # [Check] 논문 구현체 사용
        self.dtm_layer = ExactWeightedDTM(m0=m0, r=r)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = x.device
        
        # 1. Compute Exact Weighted DTM (GPU)
        # Input: [B, C, H, W] -> Output: [B, H, W]
        with torch.no_grad():
            dtm_map = self.dtm_layer(x)
        
        # GUDHI Processing (CPU)
        dtm_np = dtm_map.cpu().numpy()
        B, H, W = dtm_np.shape
        
        diagrams_batch = []
        masks_batch = []
        
        for i in range(B):
            # 2. GUDHI Cubical Complex
            # DTM 값을 Filtration value로 사용
            cc = gudhi.CubicalComplex(
                dimensions=(H, W), 
                top_dimensional_cells=dtm_np[i].flatten()
            )
            
            # 3. Persistence
            cc.persistence()
            
            # 4. Extract Intervals (Multi-dimension support)
            raw_intervals = []
            for dim in self.homology_dims:
                intervals = cc.persistence_intervals_in_dimension(dim)
                if len(intervals) > 0:
                    raw_intervals.extend(intervals)
            
            # 5. Process (Inf & Sort)
            processed_intervals = []
            for birth, death in raw_intervals:
                if np.isinf(death):
                    death = self.replace_inf_value

                processed_intervals.append([birth, death])
            
            # Sort by Lifetime
            processed_intervals.sort(key=lambda p: p[1] - p[0], reverse=True)
            
            # 6. Padding
            diag_tensor = torch.zeros((self.max_points, 2), dtype=torch.float32)
            mask_tensor = torch.zeros((self.max_points,), dtype=torch.bool)
            
            n_pts = min(len(processed_intervals), self.max_points)
            if n_pts > 0:
                pts_arr = np.array(processed_intervals[:n_pts], dtype=np.float32)
                diag_tensor[:n_pts] = torch.from_numpy(pts_arr)
                mask_tensor[:n_pts] = True
            
            diagrams_batch.append(diag_tensor)
            masks_batch.append(mask_tensor)
            
        return {
            "diagram": torch.stack(diagrams_batch).to(device),
            "mask": torch.stack(masks_batch).to(device)
        }