# CIFAR-10H: Predicting Annotator Disagreement

A two-phase ResNet-18 baseline trained to match the human annotator vote
distribution on CIFAR-10H, rather than the single hard CIFAR-10 label.

## Method

The backbone is a CIFAR-adapted ResNet-18: the standard 7x7 stride-2 stem is
replaced with a 3x3 stride-1 conv, and the first maxpool is removed so the
32x32 input is not downsampled before the residual stages. The classifier is a
two-layer MLP (512 -> 256 -> 10).

Training runs in two phases.

**Phase 1.** Pretrain on 48,000 CIFAR-10 images with cross-entropy. Adam,
lr=1e-3, weight decay=1e-4, batch size=128, up to 50 epochs with early stopping
(patience 10). The best validation-accuracy checkpoint is kept.

**Phase 2.** Fine-tune on the 6,000-image CIFAR-10H training split with
`KLDivLoss(reduction="batchmean")` against the soft label. The backbone is
frozen for the first 3 epochs to let the head adapt; then everything trains
jointly. Adam, lr=1e-4, weight decay=1e-4, `ReduceLROnPlateau` (factor 0.5,
patience 3) on validation KL. The CIFAR-10H 10,000 examples are split
6000/2000/2000 for train/val/test with seed 42.

All hyperparameters live in `src/config.py`.

## Results

CIFAR-10H test split (2,000 images). Bootstrap statistics use 1,000 resamples,
seed 42.

| Metric         | Point  | Bootstrap mean ± std |
|----------------|--------|----------------------|
| KL divergence  | 0.2288 | 0.2292 ± 0.0115      |
| JSD            | 0.0498 | 0.0498 ± 0.0023      |
| Cosine sim.    | 0.9423 | 0.9422 ± 0.0040      |
| Top-1 accuracy | 92.60% | 92.61 ± 0.59%        |

Predicted-entropy vs true-entropy correlation: Pearson r = 0.4194,
Spearman ρ = 0.4328.

High-disagreement retrieval (test images ranked by predicted entropy, compared
to the top-K by true entropy):

| K   | Precision@K |
|-----|-------------|
| 100 | 0.2300      |
| 200 | 0.3250      |
| 500 | 0.5100      |

Best validation KL during Phase 2 was 0.2135. SHA-256 hashes for the locked
checkpoint and metric files are in `results_lock_manifest.json`.

## Setup

```bash
cd project
pip install -r requirements.txt
```

Download `cifar10h-probs.npy` from
https://github.com/jcpeterson/cifar-10h and place it at
`project/data/cifar10h-probs.npy`. CIFAR-10 itself is fetched by torchvision
on first run.

Train, then evaluate:

```bash
python src/train.py        # writes best_model_phase1.pt and best_model.pt
python src/evaluate.py     # writes artifacts/evaluation_metrics.txt
```

## Reference

Peterson, Battleday, Griffiths, Yamins. *Human uncertainty makes
classification more robust.* ICCV 2019.
