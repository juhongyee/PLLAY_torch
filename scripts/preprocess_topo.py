import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import math

# 프로젝트 루트 경로 추가
sys.path.append(os.getcwd())

from src.topo.extractors import GudhiCubicalExtractor
from torchvision import datasets, transforms

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    print(">>> Start Topo Preprocessing...")
    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Extractor 초기화
    extractor = GudhiCubicalExtractor(
        max_points=cfg.model.topo.diag_max_points,
        m0=cfg.model.topo.dtm_m0,
        r=cfg.model.topo.dtm_r,
        homology_dims=list(cfg.model.topo.homology_dims),
        replace_inf_value=99.0 
    ).to(device)
    
    # 2. Raw Data Load (Torchvision 직접 사용)
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root=cfg.data.root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=cfg.data.root, train=False, download=True, transform=transform)
    
    # 저장할 폴더 생성
    save_dir = os.path.join(cfg.data.root, "processed")
    os.makedirs(save_dir, exist_ok=True)
    
    # 처리 함수
    def process_and_save(dataset, filename):
        loader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=4)
        
        all_diagrams = []
        all_masks = []
        all_labels = []
        
        max_death_val = 0.0
        
        print(f"Processing {filename}...")
        for images, labels in tqdm(loader):
            images = images.to(device)
            
            with torch.no_grad():
                out = extractor(images)
            
            diag = out["diagram"].cpu()
            mask = out["mask"].cpu()
            
            # Max Value Check
            valid_vals = diag[mask]
            if valid_vals.numel() > 0: #유효한 데이터 확인
                batch_max = valid_vals.max().item()
                if batch_max > max_death_val:
                    max_death_val = batch_max
            
            all_diagrams.append(diag)
            all_masks.append(mask)
            all_labels.append(labels)
            
        saved_data = {
            "diagrams": torch.cat(all_diagrams, dim=0),
            "masks": torch.cat(all_masks, dim=0),
            "labels": torch.cat(all_labels, dim=0)
        }
        
        save_path = os.path.join(save_dir, filename)
        torch.save(saved_data, save_path)
        print(f"Saved to {save_path}")
        print(f"Dataset Max DTM Value: {max_death_val:.4f}")
        
        return max_death_val

    # 3. 실행
    max_val_train = process_and_save(train_set, "mnist_topo_train.pt")
    max_val_test = process_and_save(test_set, "mnist_topo_test.pt")
    
    final_max = max(max_val_train, max_val_test)
    print(f"\n>>> Preprocessing Done.")
    print(f">>> [IMPORTANT] Please update 'sample_range' in config to cover: 0.0 ~ {final_max:.4f}")
    print(f">>> Recommended Config: sample_range: [0.0, {final_max + 5.0}]")

if __name__ == "__main__":
    main()