# Overall Project Output and Implementation Summary

## Project Title
Topological Data Analysis for Defending Deep Learning-Based Network Intrusion Detection Systems Against Data Poisoning Attacks

## 1. Project Overview
This project builds a defense pipeline for network intrusion detection systems (NIDS) against training-time data poisoning attacks. The main objective is to detect and remove malicious training samples before they can poison a deep learning classifier.

Rather than only improving model robustness after training, the approach targets the root cause: poisoned network payloads injected into the dataset. It combines Topological Data Analysis (TDA), clustering, dataset sanitization, and deep learning classification to create a practical security defense pipeline.

## 2. Problem Statement
Modern machine learning-based NIDS models assume the training data is trustworthy. In real adversarial environments, however, attackers can inject crafted payloads that act like triggers or backdoors during training.

These poisoned samples may look sufficiently similar to normal traffic to evade simple statistical checks, yet they interfere with the learning process and distort the final model behavior. The project addresses this by analyzing the structural geometry of payloads rather than relying only on label information or standard feature patterns.

## 3. Main Goal
The project aims to:

- detect poisoned payload samples in network traffic data,
- isolate suspicious or malicious records before training,
- clean the dataset to improve model reliability,
- train a robust deep learning-based intrusion detection model,
- evaluate whether the defense improves downstream performance.

## 4. Implementation Approach
The implementation is organized as a complete pipeline consisting of the following stages:

1. Load and preprocess the UNSW-NB15 dataset
2. Extract and normalize packet payloads
3. Inject poisoning patterns into a selected subset of samples
4. Transform payloads into binary 2D representations
5. Extract topological features using TDA
6. Use DBSCAN and HDBSCAN for suspicious cluster detection
7. Remove poisoned and highly suspicious samples
8. Train a TensorFlow NIDS on sanitized data
9. Measure final classification performance on the test set

## 5. Why TDA Is Used
Poisoned payloads often contain repeated structural patterns that are not obvious in raw byte distributions. By converting payload bytes into a binary geometric form, the project exposes hidden shape-based patterns.

Topological Data Analysis captures persistent structure across multiple scales. This makes it useful for identifying packet patterns that are structurally abnormal even when they appear superficially similar to normal traffic.

## 6. Repository and Implementation Structure
The project is organized into the following key components:

- [README.md](README.md) — project overview and reproduction steps
- [project_documentation.md](project_documentation.md) — detailed technical description
- [project_summary_ieee.md](project_summary_ieee.md) — condensed research summary
- [src](src) — implementation scripts for the full pipeline
- [data](data) — raw, processed, poisoned, sanitized, and feature datasets
- [configs](configs) — reproducibility settings
- [results](results) — evaluation metrics and training history
- [artifacts](artifacts) — trained model and preprocessing object

## 7. Script-by-Script Implementation Summary

### 01_dataset_preparation.py
Loads the UNSW-NB15 dataset, prepares a controlled experimental subset, and configures the data split used for processing and evaluation.

### 02_payload_processing.py
Extracts packet payload information and converts raw payloads into a standardized representation suitable for later geometric analysis.

### 03_poison_generation.py
Injects poisoning patterns into selected payload samples to simulate adversarial training contamination.

### 04_binarization.py
Reshapes payloads into binary 2D representations that expose the geometric and structural properties of payload bytes.

### 05_tda_features.py
Computes TDA-based topological summaries using persistent homology and related descriptors. This stage produces the feature vectors later used for clustering and suspicious-sample detection.

### 06_dbscan.py
Runs DBSCAN to cluster topological feature points and identify suspicious samples or poisoned groupings.

### 07_hdbscan.py
Runs HDBSCAN as a stronger density-based alternative for cluster detection and noise separation.

### 08_evaluation.py
Evaluates the clustering results by comparing detected suspicious samples against the poisoning labels and measuring detection metrics.

### 09_sanitize_dataset.py
Filters suspicious or poisoned samples from the training set to produce a cleaned dataset for robust model training.

### 10_deep_learning_nids.py
Trains the final deep learning intrusion detection model on the sanitized dataset and evaluates it on the held-out test set.

## 8. Dataset Used
The project uses the UNSW-NB15 benchmark dataset, which is commonly used for intrusion detection research.

Relevant files:

- [data/raw/train.csv](data/raw/train.csv)
- [data/raw/test.csv](data/raw/test.csv)

The workflow creates a stratified subset for experimentation, balancing class diversity with computational efficiency.

## 9. Poisoning Strategy
A trigger-style poisoning pattern is injected into a subset of payloads. The pattern is designed to create strong topological signatures that remain detectable even after normalization.

This is a data-poisoning simulation intended to emulate adversarial manipulation of training data in a NIDS pipeline.

## 10. Output and Results
The project reports strong evidence that TDA-based features effectively distinguish poisoned payloads from clean ones.

### Poison Detection Performance

| Method | Recall | Precision | F1-score |
|---|---:|---:|---:|
| DBSCAN | 100% | 39.1% | 56.2% |
| HDBSCAN | 100% | 83.4% | 90.9% |

Key observation:

- Both methods achieve perfect recall for poisoned samples.
- HDBSCAN performs substantially better by reducing false positives.

### Final NIDS Model Performance
The final classifier trained on sanitized data achieves the following verified results:

| Metric | Value |
|---|---:|
| Accuracy | 0.9238 |
| Precision | 0.9415 |
| Recall | 0.9390 |
| F1-score | 0.9403 |
| ROC-AUC | 0.9840 |

Confusion matrix:

| Actual \ Predicted | Normal | Attack |
|---|---:|---:|
| Normal | 1525 | 175 |
| Attack | 183 | 2818 |

These results indicate that the sanitization pipeline yields a strong and reliable intrusion detection model.

## 11. Generated Project Outputs
The execution pipeline produces the following artifact files:

- [results/deep_learning_nids_metrics.csv](results/deep_learning_nids_metrics.csv)
- [results/deep_learning_confusion_matrix.csv](results/deep_learning_confusion_matrix.csv)
- [results/deep_learning_training_history.csv](results/deep_learning_training_history.csv)
- [artifacts/deep_learning_nids_model.keras](artifacts/deep_learning_nids_model.keras)
- [artifacts/deep_learning_preprocessor.pkl](artifacts/deep_learning_preprocessor.pkl)
- [data/features/tda_features.csv](data/features/tda_features.csv)
- [data/poisoned/binarized_payloads.npy](data/poisoned/binarized_payloads.npy)

## 12. Reproduction Workflow
From the project root, the implementation can be reproduced using:

```bash
python src/01_dataset_preparation.py
python src/02_payload_processing.py
python src/03_poison_generation.py
python src/04_binarization.py
python src/05_tda_features.py
python src/06_dbscan.py
python src/07_hdbscan.py
python src/08_evaluation.py
python src/09_sanitize_dataset.py
python src/10_deep_learning_nids.py
```

## 13. Overall Conclusion
This project demonstrates that data poisoning defenses can be implemented effectively through topological feature analysis and unsupervised clustering. By identifying suspicious payload structures before training, the project removes poisoned samples and improves the quality of the downstream intrusion detection model.

The overall conclusion is that TDA-based structural analysis, combined with density-based clustering and dataset sanitization, is a promising and interpretable defense strategy for training-time adversarial attacks against deep learning-based NIDS systems.

## 14. Final Impact
The project contributes to both adversarial machine learning and cybersecurity by showing that:

- poisoned payloads can be identified using structural topological features,
- DBSCAN and HDBSCAN are effective for suspicious-sample detection,
- sanitizing poisoned training data improves model robustness,
- data-centric defense strategies are practical for real-world network security applications.
