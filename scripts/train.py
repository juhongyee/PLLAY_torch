# scripts/train.py
import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가하여 src 모듈을 찾을 수 있게 함
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.main import main

def parse_args():
    parser = argparse.ArgumentParser(description="Train PLLAY Topo-MLP")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to YAML config file"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.config)