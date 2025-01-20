# Vision Transformer Implementation

## Overview
This repository contains an implementation of the Vision Transformer (ViT) for image classification tasks, specifically on the CIFAR-10 dataset.

## Directory Structure
- `data/`: Contains datasets.
- `models/`: Model definitions.
- `scripts/`: Training, evaluation, and testing scripts.
- `configs/`: Configuration files for hyperparameters.
- `notebooks/`: Jupyter notebooks for interactive exploration.

## How to Run

### Install Dependencies:
```bash
pip install -r requirements.txt
```

### Train the Model
```bash
python scripts/train.py --config configs/config.yaml
```

### Evaluate the Model
```bash
python scripts/eval.py --config configs/config.yaml
```

### Run the Example Script
```bash
python scripts/example.py
```

### Test Script
```bash
python scripts/test.py
```
