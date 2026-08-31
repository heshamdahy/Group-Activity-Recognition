# Group Activity Recognition — Volleyball Dataset

Recognizing group activities in volleyball video clips using a two-stage, ResNet50 + hierarchical LSTM pipeline.

## Overview

This project tackles group activity recognition on the [Volleyball Dataset](https://github.com/mostafa-saad/deep-activity-rec), which contains 55 videos of volleyball rallies, each annotated with player bounding boxes and per-clip group activity labels. The approach follows a hierarchical LSTM architecture: person-level visual features are extracted per player, aggregated temporally, then pooled to reason about the group activity as a whole.

## Status

- ✅ Data pipeline: annotation parsing, bounding box loading, pickle-based caching
- ✅ First baseline: trained at the **image level**
- ⬜ Next baseline: fine-tune at the **player level**
- ⬜ Hierarchical LSTM (person-level → group-level) stage
- ⬜ Final evaluation / ablations

## Dataset

- **Source:** [Volleyball Dataset (Kaggle)](https://www.kaggle.com/datasets/ahmedmohamed365/volleyball) — 55 videos
- **Clips:** 41-frame temporal windows per annotated clip
- **Annotations:** Player bounding boxes per frame
- **Caching:** Preprocessed annotations and features are cached to disk with pickle to avoid recomputation across runs

## Pipeline

1. **Annotation loading** — parses per-video annotation files into player bounding boxes + frame indices.
2. **Feature extraction** — ResNet50 (pretrained, backbone frozen/fine-tuned as configured) extracts visual features across the 41-frame window.
3. **First baseline (image level)** — a baseline model trained on image-level features to establish an initial group activity classification benchmark.
4. **(Next) Player-level fine-tuning** — fine-tune the model at the individual player level (per-player crops/features) rather than the whole image, as a stronger baseline before moving to the hierarchical LSTM.
5. **(Later) Hierarchical LSTM** — a two-stage LSTM: stage 1 models individual player temporal dynamics, stage 2 aggregates player representations into a group-level temporal model for final classification.

## First Baseline (B1) — Details

Following the standard B1 image-classification baseline approach:

- Fine-tune an image classifier (ResNet50) over the 8 group-activity classes
- For each clip, use the **middle frame only** (optionally extend to the surrounding frames, e.g. 4 before / 5 after)
- No temporal modeling at this stage — plain per-image fine-tuning
- This model is the starting point before moving to person-level and temporal baselines

### Results

| Metric | Value | Step |
|---|---|---|
| Train Accuracy | 95.44% | 89 |
| Val Accuracy | 54.43% (best run reached 57.21%) | 89 / 0 |
| Train Loss | 0.0689 | 89 |

**Train Accuracy**

![Train Accuracy](train_accuracy.png)

**Validation Accuracy**

![Validation Accuracy](val_accuracy.png)

**Train Loss**

![Train Loss](train_loss.png)

> Train accuracy is high while val accuracy is much lower and noisy — a sign of overfitting at the image level, which the player-level and temporal baselines below aim to address.

### Reference Benchmarks (from the paper)

| Method | Accuracy |
|---|---|
| B1 – Image Classification | 66.7 |
| B2 – Person Classification | 64.6 |
| B3 – Fine-tuned Person Classification | 68.1 |
| B4 – Temporal Model with Image Features | 63.1 |
| B5 – Temporal Model with Person Features | 67.6 |
| B6 – Two-stage Model without LSTM 1 | 74.7 |
| B7 – Two-stage Model without LSTM 2 | 80.2 |
| **Two-stage Hierarchical Model** | **81.9** |

## Tools
-python
-pytorch

## Next Steps

- Fine-tune at the player level (next baseline)
- Implement the hierarchical (two-stage) LSTM architecture
- Compare hierarchical LSTM performance against the baselines
- Log and report evaluation metrics (accuracy, confusion matrix per activity class)

## Reference

- Ibrahim et al., *A Hierarchical Deep Temporal Model for Group Activity Recognition* (CVPR 2016)
