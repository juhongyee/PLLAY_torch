# src/models/topo_mlp.py
from __future__ import annotations
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from src.config import ModelConfig
from src.models.components import MLP
from src.topo.dummy import DummyTopoExtractor, DummyTopoEmbedder

# 나중에 실제 구현체가 생기면 import 교체
# from src.topo.extractors import GudhiExtractor
# from src.topo.embedders import LandscapeEmbedder

class TopoMLPClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_classes = cfg.num_classes
        
        # 1. Image Backbone (MLP)
        # 이미지(28x28)를 Flatten해서 넣는다고 가정
        self.image_backbone = MLP(
            in_dim=cfg.image_backbone.in_dim,       # 784
            hidden_dims=cfg.image_backbone.hidden_dims,
            out_dim=cfg.image_backbone.out_dim,     # D_x
            dropout=cfg.image_backbone.dropout
        )
        self.image_out_dim = cfg.image_backbone.out_dim

        # 2. Topo Branch
        # 설정에 따라 Dummy를 쓸지, 실제를 쓸지, 아예 안 쓸지 결정
        self.use_topo = (cfg.topo.out_dim > 0)
        self.topo_out_dim = 0
        
        if self.use_topo:
            self.topo_out_dim = cfg.topo.out_dim
            
            # Config의 max_points를 반영하여 Dummy 초기화
            self.topo_extractor = DummyTopoExtractor(
                max_points=cfg.topo.diag_max_points
            ) 
            self.topo_embedder = DummyTopoEmbedder(out_dim=self.topo_out_dim)
            
        # 3. Fusion & Head
        # Concat(Image, Topo) -> MLP Head -> Logits
        fusion_dim = self.image_out_dim + self.topo_out_dim
        
        self.head = MLP(
            in_dim=fusion_dim,
            hidden_dims=cfg.head.hidden_dims,
            out_dim=self.num_classes, # K
            dropout=cfg.head.dropout
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Contract:
          Input batch: {"image": [B,C,H,W], ...}
          Output: {"logits": [B,K], "image_feat": ..., "topo_feat": ...}
        """
        x = batch["image"] # [B, C, H, W]
        B = x.shape[0]

        # 1. Image Feature
        # Flatten: [B, C, H, W] -> [B, C*H*W]
        x_flat = x.view(B, -1)
        image_feat = self.image_backbone(x_flat) # [B, D_x]

        # 2. Topo Feature
        topo_feat = None
        
        if self.use_topo:
            # Case A: 데이터셋이 이미 topo feature를 제공하는 경우 (Offline Mode)
            if "topo" in batch and "topo_feat" in batch["topo"]:
                topo_feat = batch["topo"]["topo_feat"]
            
            # Case B: 모델이 직접 계산해야 하는 경우 (Online Mode / Dummy Test)
            else:
                # 지금은 Dummy라 랜덤값 나옴
                topo_data = self.topo_extractor(x)
                topo_feat = self.topo_embedder(topo_data, batch_size=B, device=x.device)

        # 3. Fusion
        if topo_feat is not None:
            # [B, D_x] + [B, D_t] -> [B, D_x + D_t]
            combined = torch.cat([image_feat, topo_feat], dim=1)
        else:
            combined = image_feat

        # 4. Head
        logits = self.head(combined) # [B, K]

        return {
            "logits": logits,
            "image_feat": image_feat,
            "topo_feat": topo_feat # 디버깅용 (None일 수 있음)
        }