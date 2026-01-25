# src/data/transforms.py
from __future__ import annotations
import torch
import torch.nn as nn

class CorruptAndNoise(nn.Module):
    """
    Apply corruption and noise to images.
    
    Args:
        corrupt_prob: Probability to drop a pixel to 0 (Corruption).
        noise_prob: Probability to add uniform noise to background pixels.
        background_threshold: Value threshold to consider a pixel as 'background'.
                              Default 0.01 (for [0,1] images). Use ~2.55 for [0,255].
        noise_high: Upper bound for uniform noise. 
                    Default 1.0 (for [0,1] images). Use 255.0 for [0,255].
    """
    def __init__(
        self, 
        corrupt_prob: float, 
        noise_prob: float, 
        background_threshold: float = 0.01,
        noise_high: float = 1.0
    ):
        super().__init__()
        self.corrupt_prob = corrupt_prob
        self.noise_prob = noise_prob
        self.background_threshold = background_threshold
        self.noise_high = noise_high

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: FloatTensor [C, H, W]
        """
        # 1. Corruption (Dropout pixels)
        # 픽셀을 0으로 만드는 것은 스케일과 무관하게 동작
        if self.corrupt_prob > 0:
            # keep_prob = 1 - corrupt_prob
            mask = torch.bernoulli(torch.full_like(img, 1 - self.corrupt_prob))
            img = img * mask

        # 2. Noise (Background Noise)
        # 배경 인식과 노이즈 값은 스케일에 의존하므로 파라미터 사용
        if self.noise_prob > 0:
            # 배경 마스크
            is_background = (img < self.background_threshold)
            
            # 노이즈를 추가할 픽셀 선택
            noise_mask = torch.bernoulli(torch.full_like(img, self.noise_prob)).bool()
            
            # 배경이면서 & 노이즈 당첨된 픽셀
            target_pixels = is_background & noise_mask
            
            # Uniform Noise [0, noise_high] 생성
            noise_values = torch.rand_like(img) * self.noise_high
            
            # 원본 이미지에 덮어쓰기
            img = torch.where(target_pixels, noise_values, img)

        return img

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(corrupt={self.corrupt_prob}, "
                f"noise={self.noise_prob}, bg_thresh={self.background_threshold})")