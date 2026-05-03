# Predicting Human Annotator Disagreement on CIFAR-10H

A two-phase PyTorch baseline that learns to predict the *distribution* of human annotator votes on CIFAR-10 images, rather than a single hard label.

---

## Problem Statement

Human annotators disagree on ambiguous images. CIFAR-10H captures this as soft-label probability distributions over the 10 CIFAR-10 classes. This project trains a model whose output distribution matches those human-disagreement patterns, using KL divergence as the training objective.

---

## Dataset

| Dataset | Split | Size | Labels |
|---|---|---|---|
| CIFAR-10 | train | 50,000 | Hard labels (one-hot) |
| CIFAR-10H | train / val / test | 6,000 / 2,000 / 2,000 | Soft labels (annotator vote distributions) |

CIFAR-10H consists of soft labels over the standard 10,000-image CIFAR-10 test set. The 6000/2000/2000 split uses a fixed seed (42) for reproducibility.

Download `cifar10h-probs.npy` from: https://github.com/jcpeterson/cifar-10h  
Place it at: `project/data/cifar10h-probs.npy`  
CIFAR-10 images are downloaded automatically by torchvision.

---

## Method

### Architecture

CIFAR-adapted ResNet-18:
- **Stem**: 3×3 conv, stride 1 (replaces the standard 7×7 stride-2 stem to preserve 32×32 detail)
- **Maxpool**: replaced with Identity (no early downsampling)
- **Head**: 2-layer MLP — Linear(512→256) → ReLU → Linear(256→10)
- **Output**: raw logits; softmax/log_softmax applied in loss/eval code

### Two-Phase Training

**Phase 1 — CIFAR-10 pretraining**  
Trains on 48,000 CIFAR-10 hard-label examples (2,000 held out for validation) using CrossEntropy loss. Checkpoint criterion: best validation accuracy.

**Phase 2 — CIFAR-10H fine-tuning**  
Loads the Phase 1 checkpoint and fine-tunes on CIFAR-10H soft labels using KL divergence loss (`KLDivLoss(reduction="batchmean")`). The backbone is frozen for the first 3 epochs to stabilize the head before joint training. A `ReduceLROnPlateau` scheduler monitors validation KL. Checkpoint criterion: best validation KL.

### Hyperparameters

| Parameter | Phase 1 | Phase 2 |
|---|---|---|
| Optimizer | Adam | Adam |
| Learning rate | 1e-3 | 1e-4 |
| Weight decay | 1e-4 | 1e-4 |
| Batch size | 128 | 128 |
| Max epochs | 50 | 50 |
| Early stopping patience | 10 | 10 |
| Backbone freeze | — | First 3 epochs |
| LR scheduler | — | ReduceLROnPlateau (factor=0.5, patience=3) |

---

## Setup

```bash
cd project
pip install -r requirements.txt
```

Place `cifar10h-probs.npy` in `project/data/` (see Dataset section above).

---

## Training

```bash
python src/train.py
```

Saves `best_model_phase1.pt` and `best_model.pt` to `project/`, and appends epoch logs to `training_log.csv`.

> **Note:** Running training overwrites the locked output files. See `results_lock_manifest.json` for the SHA-256 hashes of the final locked results.

---

## Evaluation

Evaluate the fine-tuned model on the CIFAR-10H test split:

```bash
python src/evaluate.py
```

`evaluate.py` reports point estimates and bootstrap mean ± std (n=1000) for KL / JSD / Cosine / Top-1, plus Pearson and Spearman correlations between predicted and true entropy and Precision@K (K = 100, 200, 500) for high-disagreement retrieval. Results are written to `project/artifacts/evaluation_metrics.txt`. Test-split softmax outputs and targets are also cached as `artifacts/test_predictions.npy` and `artifacts/test_targets.npy` so downstream plots (e.g., the entropy scatter) can be regenerated without re-running inference.

---

## Artifact Generation

Generate all analysis artifacts without retraining:

```bash
python src/generate_artifacts.py        # full artifact suite
python src/generate_artifacts.py --training-only  # training curves only
```

Run ablation studies (requires re-training ablation variants):

```bash
python src/compare_heads.py     # linear vs MLP-2 vs MLP-3 head
python src/compare_losses.py    # KL vs JSD vs Soft-CE loss
python src/robustness_eval.py   # clean / noise / blur / brightness
python src/gradcam_analysis.py  # Grad-CAM for high- and low-entropy examples
```

---

## Results

### Main model (CIFAR-10H test split)

Point estimates and bootstrap mean ± std (n=1000 resamples) over the 2,000-sample test split:

| Metric | Value | Bootstrap mean ± std |
|---|---|---|
| KL Divergence | 0.2288 | 0.2292 ± 0.0115 |
| JSD | 0.0498 | 0.0498 ± 0.0023 |
| Cosine Similarity | 0.9423 | 0.9422 ± 0.0040 |
| Top-1 Accuracy | 92.60% | 92.61 ± 0.59% |

Best validation KL (Phase 2): **0.2135** (recorded in `results_lock_manifest.json`).

### Predicted vs true entropy

| Metric | Value |
|---|---|
| Pearson r (predicted-entropy vs true-entropy) | 0.4194 |
| Spearman ρ | 0.4328 |

Scatter plot: `artifacts/entropy_scatter.png`.

### High-disagreement retrieval (Precision@K)

Test images ranked by predicted entropy, compared against the top-K by true entropy:

| K | Precision@K |
|---|---|
| 100 | 0.2300 |
| 200 | 0.3250 |
| 500 | 0.5100 |

### Prediction head ablation (separate training runs)

| Head | KL | JSD | Cosine | Acc |
|---|---|---|---|---|
| Linear | 0.312 | 0.065 | 0.919 | 89.20% |
| MLP-2 | 0.322 | 0.073 | 0.909 | 88.65% |
| MLP-3 | 0.333 | 0.071 | 0.911 | 89.20% |

### Loss function ablation (separate training runs)

| Loss | KL | JSD | Cosine | Acc |
|---|---|---|---|---|
| KL | 0.320 | 0.071 | 0.912 | 88.70% |
| JSD | 0.426 | 0.063 | 0.917 | 89.85% |
| Soft CE | 0.329 | 0.070 | 0.908 | 88.35% |

### Robustness (main model)

| Condition | KL | Top-1 Acc |
|---|---|---|
| Clean | 0.229 | 92.60% |
| Gaussian noise | 2.384 | 30.20% |
| Gaussian blur | 2.008 | 38.95% |
| Brightness shift | 0.247 | 91.50% |

---

## Artifacts

| File | Description |
|---|---|
| `artifacts/architecture_diagram.png` | Block diagram of the CIFAR-adapted ResNet-18 |
| `artifacts/evaluation_metrics.txt` | Test-split KL, JSD, cosine, top-1, bootstrap std, entropy correlations, Precision@K |
| `artifacts/entropy_scatter.png` | Predicted vs true entropy scatter (test split) |
| `artifacts/test_predictions.npy` | Cached softmax outputs for the test split (used by entropy scatter) |
| `artifacts/test_targets.npy` | Cached soft-label targets for the test split |
| `artifacts/training_loss_curve.png` | Train loss vs epoch (Phase 1 + Phase 2) |
| `artifacts/validation_loss_curve.png` | Validation loss vs epoch |
| `artifacts/phase2_kl_curve.png` | Phase 2 train/val KL per epoch |
| `artifacts/entropy_histogram.png` | CIFAR-10H soft-label entropy distribution |
| `artifacts/per_class_average_entropy.png` | Mean entropy grouped by majority class |
| `artifacts/annotator_confusion_matrix.png` | Average soft-label mass conditioned on majority class |
| `artifacts/low_high_entropy_examples.png` | Image grid — lowest vs highest-entropy examples |
| `artifacts/model_summary.txt` | Parameter counts by layer group |
| `artifacts/data_summary.txt` | Entropy statistics and split sizes |
| `artifacts/failure_analysis.txt` | Failure case analysis |
| `artifacts/head_comparison.txt` | Head ablation results table |
| `artifacts/loss_comparison.txt` | Loss ablation results table |
| `artifacts/robustness.txt` | Robustness evaluation results |
| `artifacts/gradcam/` | Grad-CAM visualizations (high/low entropy) |
| `artifacts/head_compare/` | Checkpoints and logs for head ablation runs |
| `artifacts/loss_compare/` | Checkpoints and logs for loss ablation runs |

---

## Repository Layout

```
project/
├── src/
│   ├── config.py              # Hyperparameters and paths
│   ├── dataset.py             # CIFAR-10 / CIFAR-10H dataloaders
│   ├── model.py               # CIFAR-adapted ResNet-18
│   ├── train.py               # Two-phase training loop
│   ├── generate_artifacts.py  # Post-training analysis and plots
│   ├── evaluate.py            # Test-split evaluation metrics
│   ├── compare_heads.py       # Prediction head ablation
│   ├── compare_losses.py      # Loss function ablation
│   ├── robustness_eval.py     # Robustness under input corruptions
│   └── gradcam_analysis.py    # Grad-CAM and failure analysis
├── artifacts/                 # Generated plots, tables, and checkpoints
├── data/                      # CIFAR-10 cache and cifar10h-probs.npy
├── best_model.pt              # Final fine-tuned checkpoint (Phase 2)
├── best_model_phase1.pt       # Phase 1 pretraining checkpoint
├── training_log.csv           # Epoch-level loss log (epoch,phase,train_loss,val_loss)
├── results_lock_manifest.json # SHA-256 integrity hashes for locked outputs
└── requirements.txt           # Python dependencies
```

---

## Reproducibility

- All experiments use a fixed random seed (42) for data splits and weight initialization.
- Device is resolved automatically: CUDA → MPS → CPU.
- The five locked output files are pinned with SHA-256 hashes in `results_lock_manifest.json`.
- Bootstrap statistics use a fixed seed (42) and 1,000 resamples of the 2,000-sample test split.

---

## Known Gaps

The following items from the project specification are **not** addressed in the current artifacts. They require additional training runs that were out of scope for this final cleanup pass (no retraining was performed). All cleanup-pass additions above (Pearson/Spearman correlation, Precision@K, bootstrap std, entropy scatter, architecture diagram) are computed from the locked Phase-2 checkpoint and are reproducible by running `evaluate.py` followed by `generate_artifacts.py`.

- **Custom / composite loss.** The loss ablation covers KL, JSD, and Soft-CE — three standard divergence-style objectives. A task-specific composite (e.g., KL + entropy-error penalty, focal-weighted KL, or a rank-aware probability penalty) is not implemented.
- **Third ablation track.** Two of the four ablation tracks listed in the spec (Loss, Head) are reported; the spec asks for at least three, so one of {Backbone Init, Training-Data Strategy} is missing.
- **Second robustness study.** Only the OOD/corruption track is reported. Annotator-subsampling and class-conditional robustness are not.
- **Corruption-severity sweep plot.** The robustness study reports four discrete conditions (clean, noise, blur, brightness) as a table; no continuous severity-vs-metric curve is plotted. The spec also lists "contrast reduction" as an example whereas the implemented code uses a brightness shift.
- **Manual disagreement-source categorization.** `failure_analysis.txt` includes Grad-CAM-based interpretation notes for the 5 lowest- and 5 highest-entropy samples, but the spec's manual taxonomy (ambiguous identity / poor quality / multi-label / boundary case / other) is not applied.

---

## References

- Peterson, J. C., Battleday, R. M., Griffiths, T. L., & Yamins, D. L. K. (2019). *Human uncertainty makes classification more robust.* ICCV. [CIFAR-10H]
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition.* CVPR. [ResNet]
