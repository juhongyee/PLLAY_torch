# 아직 code review 안했음.

import argparse
import sys
import dataclasses
from pathlib import Path
import torch
from tqdm import tqdm

# 프로젝트 루트 추가
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import load_config
from src.utils.device import get_device
from src.utils.seed import set_seed
from src.data.dataloader import build_dataloader
from src.models.topo_mlp import TopoMLPClassifier

def evaluate_model(model, loader, device, amp_enabled=False):
    """
    단순 평가 루프 (Trainer.evaluate와 유사하지만 Standalone)
    """
    model.eval()
    correct = 0
    total = 0
    
    # AMP Scaler는 평가 때 불필요하지만, autocast는 씀
    from torch.cuda.amp import autocast

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            # Move to device
            img = batch["image"].to(device)
            target = batch["label"].to(device)
            
            # Topo Feature도 있으면 이동
            if "topo" in batch and "topo_feat" in batch["topo"]:
                batch["topo"]["topo_feat"] = batch["topo"]["topo_feat"].to(device)

            # Forward
            with autocast(enabled=amp_enabled):
                # 모델 입력은 dict 전체를 넘김
                # (주의: Dataset에서 dict로 줬으므로 배치 텐서들도 dict 안에 있음)
                # 하지만 현재 batch 변수는 collate된 dict임.
                # 모델은 batch dict를 통째로 받도록 설계됨.
                # 단, batch 안의 텐서들은 device로 옮겨야 함.
                
                # 재귀적으로 device 이동이 귀찮으니, 필요한 것만 담아서 넘기거나
                # Trainer의 _to_device 로직을 가져와야 함.
                # 여기서는 간단히 dict 컴프리헨션 사용
                model_input = {}
                model_input["image"] = img
                model_input["label"] = target
                if "id" in batch: model_input["id"] = batch["id"]
                if "topo" in batch: model_input["topo"] = batch["topo"]

                outputs = model(model_input)
                logits = outputs["logits"]

            preds = logits.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)

    acc = correct / total
    return acc

def main(args):
    # 1. Config 로드
    cfg = load_config(args.config)
    
    # 2. Config Override (노이즈 설정 변경)
    # frozen=True 이므로 replace 사용
    new_augment = dataclasses.replace(
        cfg.data.augmentation,
        corrupt_prob=args.corrupt_prob,
        noise_prob=args.noise_prob
    )
    new_data = dataclasses.replace(cfg.data, augmentation=new_augment)
    cfg = dataclasses.replace(cfg, data=new_data)
    
    print(f"[Info] Evaluation Config:")
    print(f"  - Corrupt Prob: {cfg.data.augmentation.corrupt_prob}")
    print(f"  - Noise Prob:   {cfg.data.augmentation.noise_prob}")
    print(f"  - Model Path:   {args.checkpoint}")

    # 3. Setup
    dev_info = get_device(cfg.device)
    set_seed(cfg.seed)
    
    # 4. Data Loader (Test Split)
    # 평가용이므로 shuffle=False
    loader = build_dataloader(cfg.data, split="test", shuffle=False)
    
    # 5. Model Build
    model = TopoMLPClassifier(cfg.model)
    model.to(dev_info.device)
    
    # 6. Load Checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=dev_info.device)
    
    # state_dict 키 매칭 (module. 접두사 처리 등)
    state_dict = checkpoint["model_state_dict"]
    # 만약 DataParallel로 저장되었다면 키 앞에 'module.'이 붙어있을 수 있음 -> 제거 로직 필요하면 추가
    
    model.load_state_dict(state_dict)
    
    # 7. Run Evaluation
    acc = evaluate_model(model, loader, dev_info.device, cfg.train.amp)
    print(f"\n>>> Test Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Original training config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model.pt or best_model.pt")
    
    # Robustness Test를 위한 오버라이드 인자
    parser.add_argument("--corrupt_prob", type=float, default=0.0, help="Override corruption prob")
    parser.add_argument("--noise_prob", type=float, default=0.0, help="Override noise prob")
    
    args = parser.parse_args()
    main(args)