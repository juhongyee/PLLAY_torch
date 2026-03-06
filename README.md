# PLLAY_torch
PyTorch implementation of PLLAY.

## Repository Structure

```text
PLLAY_torch/
├── configs/          # experiment config
├── scripts/          # preprocess / train / eval scripts
├── src/              # core source code
├── data/             # dataset and processed topology features
├── runs/             # training outputs
└── outputs/          # saved outputs
```

## Quick Start

### 1. Install dependencies
Create a virtual environment and install the required packages.

```bash
pip install -r requirements.txt
```

### 2. Precompute topological features
This step generates preprocessed topology files for MNIST.

```bash
python scripts/preprocess_topo.py
```

### 3. Train
```bash
python scripts/train.py --config configs/default.yaml
```

### 4. Evaluate
```bash
python scripts/eval.py --config configs/default.yaml --checkpoint runs/mnist_topo_pllay_001/best_model.pt
```

# Notes
- The default config is `configs/default.yaml`.
- Preprocessed files are expected at:
  - `data/processed/mnist_topo_train.pt`
  - `data/processed/mnist_topo_test.pt`
- Training outputs are saved under `runs/${run_name}`.