# src/utils/device.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    is_cuda: bool
    cuda_index: Optional[int]
    name: Optional[str]

def get_device(device_str: str = "cuda") -> DeviceInfo:
    """
    device_str을 파싱하여 사용할 디바이스를 반환합니다.
    Fallback: 요청한 CUDA 디바이스가 없으면 자동으로 CPU를 반환합니다.
    Examples: "auto", "cuda", "cuda:1", "cpu"
    """
    if not isinstance(device_str, str) or not device_str.strip():
        raise ValueError("device_str must be a non-empty string")
    
    s = device_str.strip().lower()

    # Auto or CUDA requested
    if s == "auto" or s.startswith("cuda"):
        if not torch.cuda.is_available():
            # Fallback to CPU
            return DeviceInfo(
                device=torch.device("cpu"), 
                is_cuda=False, 
                cuda_index=None, 
                name=None
            )
        
        # CUDA is available
        idx = 0
        if ":" in s:
            parts = s.split(":")
            if len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
        
        # Validate index range
        if idx >= torch.cuda.device_count():
            # Index out of range -> Fallback to cuda:0
            idx = 0
        
        dev = torch.device(f"cuda:{idx}")
        name = torch.cuda.get_device_name(idx)
        return DeviceInfo(device=dev, is_cuda=True, cuda_index=idx, name=name)

    # CPU requested
    if s == "cpu":
        dev = torch.device("cpu")
        return DeviceInfo(device=dev, is_cuda=False, cuda_index=None, name=None)

    raise ValueError(f"Unsupported device string: {device_str}")