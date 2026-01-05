# Melting Point Prediction Challenge

## Description
Predicting the **melting point** of organic molecules is a long-standing challenge in chemistry and chemical engineering. Melting point is critical for drug design, material selection, and process safety, yet experimental measurements are often costly, time-consuming, or unavailable.

In this competition, you’ll use **machine learning** to predict melting points from **group contribution features**, subgroup counts that represent functional groups within each molecule. Your task is to build models that capture the complex, nonlinear relationships between molecular structure and melting behavior.

## Metric
Submissions are scored using **Mean Absolute Error (MAE)** on a held-out test set. Lower is better.

## Dataset Description
- **Total compounds**: 3328
- **Train**: 2662 (80%)
- **Test**: 666 (20%)

### Files
- `train.csv`: Features + target (Tm)
- `test.csv`: Features only, no target
- `sample_submission.csv`: Template with columns [id, Tm]
