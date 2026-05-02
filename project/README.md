# Predicting Human Annotator Disagreement (CIFAR-10H)

Minimal two-phase PyTorch baseline:
1. Pretrain on CIFAR-10 hard labels.
2. Fine-tune on CIFAR-10H soft labels.

## Scope Files
- `src/config.py` - hyperparameters and paths
- `src/dataset.py` - CIFAR-10/CIFAR-10H loading + dataloaders
- `src/model.py` - CIFAR-adapted ResNet-18 + 2-layer head
- `src/train.py` - two-phase training loop + logging

## Quick Start
```bash
cd project
pip install -r requirements.txt
```

Download `cifar10h-probs.npy` from:
https://github.com/jcpeterson/cifar-10h

Place it at:
```text
project/data/cifar10h-probs.npy
```

CIFAR-10 images are downloaded automatically by torchvision.

Run training:
```bash
python src/train.py
```

Generate required artifacts (no retraining):
```bash
python src/generate_artifacts.py
```

Run evaluation (CIFAR-10H test split):
```bash
python src/evaluate.py
```

Metrics are saved to:
```text
project/artifacts/evaluation_metrics.txt
```

## Outputs (saved in `project/`)
- `best_model_phase1.pt`
- `best_model.pt`
- `training_log.csv` (`epoch,phase,train_loss,val_loss`)
