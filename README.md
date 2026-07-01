# RobustKNNGraphEnsemble

This folder contains the Python code needed to reproduce the proposed Paper 1 method only. It intentionally excludes benchmark comparisons, result tables, and plotting code.

The method is a train-only similarity-graph ensemble for multiclass intrusion detection. It builds k-nearest-neighbor graph features from training records, appends neighbor class-probability and entropy features, augments the training data with small numeric perturbations, and fits a soft-voting Random Forest, ExtraTrees, and XGBoost classifier.

## Repository Contents

- `src/robust_knn_graph_ensemble.py`: implementation of the proposed estimator.
- `src/data_loading.py`: dataset loading, label normalization, leakage-column removal, and split logic used for the selected Paper 1 datasets.
- `run_reproduce_method.py`: minimal runner for training and evaluating only the proposed method.
- `requirements.txt`: Python dependencies.

## Data

The runner expects the same local dataset layout used in the paper experiments under a single data root. For the original experiments, the data root was:

```text
C:\Users\MMASSAOUDI\Desktop\Data
```

The supported dataset names are:

- `CICIoT23_multiclass`
- `EdgeIIoT_multiclass`
- `RT_IOT2022_attack_type`

## Installation

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Reproduce The Method

Run the proposed method on all three datasets with seed 42:

```bash
python run_reproduce_method.py --data-root "C:\Users\MMASSAOUDI\Desktop\Data" --dataset all --seed 42 --output method_results.json
```

Run the proposed method on one dataset:

```bash
python run_reproduce_method.py --data-root "C:\Users\MMASSAOUDI\Desktop\Data" --dataset EdgeIIoT_multiclass --seed 42 --output method_results.json
```

The output is a JSON file containing only the proposed-method metrics. No benchmark comparison, paper table, or figure is generated.

## Method Summary

The estimator follows these steps:

1. Clean column names and replace infinite values with missing values.
2. Encode class labels inside the estimator.
3. Fit a numeric imputer and scaler on training records only.
4. Fit a k-nearest-neighbor index on scaled training numeric features only.
5. For every training and test record, append train-only neighbor class-probability features and a normalized entropy feature.
6. Duplicate the graph-augmented training set with small Gaussian numeric perturbations.
7. Fit a preprocessing pipeline for numeric and categorical features.
8. Train a soft-voting ensemble using Random Forest, ExtraTrees, and XGBoost when XGBoost is installed.

## Leakage Controls

The test set is never used to fit the k-nearest-neighbor graph index, imputers, scalers, encoders, or classifiers. Edge-IIoTset leakage-prone columns such as labels, IP addresses, timestamps, payloads, messages, URIs, and DNS queries are removed before fitting.

