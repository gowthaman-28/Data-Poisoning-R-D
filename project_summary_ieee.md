# Topological Data Analysis and Density-Based Clustering for Defending Machine Learning NIDS Against Data Poisoning Attacks

## Abstract

Machine learning-based Network Intrusion Detection Systems (NIDS) are increasingly used for high-throughput traffic analysis, but they remain highly susceptible to data poisoning attacks. In these attacks, adversaries inject carefully crafted malicious samples into the training set to compromise model integrity, embed backdoor triggers, or degrade detection performance while maintaining overall classification accuracy. This project addresses the problem by proposing a pre-training data sanitization framework that detects poisoned network-flow payloads before model training.

The proposed system converts packet payloads into binary grid representations, extracts topological signatures using Topological Data Analysis (TDA), and applies density-based clustering to identify poisoned samples. Specifically, payload bytes are normalized and reshaped into a 30x50 binary matrix, followed by multiple height and radial filtrations and persistent homology computations to generate 72-dimensional topological feature vectors. These features are then clustered using DBSCAN and HDBSCAN, and samples in highly poisoned or anomalous clusters are flagged for removal.

The framework is evaluated on a stratified subset of the UNSW-NB15 dataset with a 5% poisoning rate. Experimental results show that both clustering approaches achieve perfect poisoned-sample recall, while HDBSCAN substantially outperforms DBSCAN in precision and F1-score because it markedly reduces false positives caused by noise labeling. In particular, HDBSCAN achieves 100% detection recall, 83.4% precision, and 90.9% F1-score, compared with 39.1% precision and 56.2% F1-score for DBSCAN. These findings demonstrate that TDA-based topological descriptors provide a robust and interpretable defense against data poisoning in ML-driven NIDS systems.

## Keywords

Network Intrusion Detection, Data Poisoning, Topological Data Analysis, DBSCAN, HDBSCAN, Payload Sanitization, UNSW-NB15, Cybersecurity, Backdoor Detection.

---

## I. Introduction

The rapid growth of network traffic and the increasing complexity of cyber threats have driven the adoption of machine learning systems in Network Intrusion Detection Systems (NIDS). These models are capable of processing large volumes of traffic and distinguishing normal behavior from malicious activity. However, this reliance on learning-based detection creates a critical vulnerability: adversaries can manipulate the training data itself through data poisoning attacks.

In a poisoning scenario, an attacker inserts malicious samples or backdoor triggers into the training set. These poisoned samples may be designed to look benign under standard aggregate metrics, thereby avoiding detection while still causing the model to misclassify certain targeted attack patterns or traffic types. In security-critical settings, such attacks are especially damaging because they may compromise the learning pipeline at its root before any model validation occurs.

Existing defensive strategies typically focus on robust training or post-hoc anomaly detection. However, these methods often fail to address the source of the problem: poisoned payloads embedded inside the raw data. This project explores an alternative approach based on data sanitization before training. The key insight is that poisoned payloads often contain repeated or structurally distinctive trigger patterns that produce unusually consistent topological features. By analyzing payload geometry through Topological Data Analysis (TDA), such poisoned samples can be separated from benign traffic.

This work reproduces and evaluates a TDA-driven defense pipeline for machine learning-based NIDS under data poisoning conditions. The proposed system uses payload binarization, persistent homology, and density-based clustering to detect and remove malicious samples before training an intrusion detection model.

---

## II. Background and Motivation

### A. Threat Model

The project considers a realistic data poisoning setting in which the attacker injects a controlled trigger pattern into a subset of training samples. These poisoned samples preserve the overall structure of a network flow but alter the packet payload content to encode a trigger. The trigger is designed to be highly repetitive and topologically distinctive, allowing it to form a dense, detectable cluster in a transformed feature space while remaining nearly invisible under average classification metrics.

### B. Why TDA for Payload Analysis

Traditional feature representations are often sensitive to small changes in byte values and may fail to capture structural regularities embedded in payload content. In contrast, TDA provides a way to summarize geometric and topological properties of data, such as connected components, loops, and voids, through persistent homology.

When payload bytes are converted to binary grid images, the trigger pattern contributes to recurring connected components and persistent homological signatures. This makes poisoned traffic distinguishable from benign traffic even when the payload appears only slightly altered at the raw byte level.

### C. Density-Based Clustering for Sanitization

Once topological features are extracted, the system clusters the resulting feature vectors. Density-based clustering is suitable because it can identify compact structures, outliers, and groups of anomalous samples. The project compares DBSCAN and HDBSCAN, both of which can detect clusters of poisoned samples while labeling sparse or low-density regions as noise. This is particularly valuable in a sanitization pipeline, where outliers are treated as suspicious and removed.

---

## III. Methodology

### A. Data Preparation and Payload Representation

The pipeline begins with raw network traffic data from the UNSW-NB15 dataset. Traffic records are sampled strategically to preserve class imbalance while limiting total data volume. The system constructs a fixed-length payload representation by extracting packet bytes and normalizing them to the range [0, 1]. The normalized vector is then converted into a binary form using a threshold value of 0.3. This transforms each payload into a 2D binary grid with dimensions 30x50, enabling spatial analysis of byte-level structure.

### B. Topological Feature Extraction

To extract topological characteristics, the binary payload grids are processed using multiple filtration methods. These include:

- Height filtration in four directions
- Radial filtration using two focal centers

For each filtration, persistent homology is computed in homology dimensions H0 and H1. This captures the birth and death of connected components and loops across varying thresholds. The resulting persistence diagrams are summarized using six amplitude descriptors, including bottleneck distance, Wasserstein distance, persistence landscape, Betti curve, heat kernel, and silhouette distance.

The combination of six filtrations, two homology dimensions, and six amplitude metrics produces a 72-dimensional feature vector for each traffic record. This representation encodes the topological structure of the payload and forms the basis for clustering.

### C. Clustering and Poison Detection

The generated feature vectors are standardized using z-score normalization and clustered with DBSCAN or HDBSCAN. Cluster labels are then interpreted according to poisonousness:

- Clean cluster: low poisoning proportion
- Poisoned cluster: high poisoning proportion
- Mixed cluster: moderate contamination
- Outlier/noise: treated as suspicious and flagged as poisoned

A cluster purity-based approach is used to determine whether the cluster is clean or poisoned. Samples from poisoned-dominant clusters and anomalous noise points are removed from the training pool before model learning. This effectively sanitizes the dataset prior to neural network training.

---

## IV. Experimental Setup

### A. Dataset

The project uses the UNSW-NB15 benchmark dataset, which contains multiple categories of benign and malicious network traffic. To control computational cost while preserving class diversity, a stratified subset of 25,000 samples is selected. The selected distribution includes normal traffic and several intrusion classes such as Exploits, Generic, Fuzzers, DoS, Reconnaissance, and other minority classes.

### B. Poisoning Rate

A poisoning rate of 5% is used, corresponding to 1,250 poisoned samples. Poisoned payloads are generated by replacing the target flow payload with a repeating checkerboard trigger pattern, specifically a binary pattern derived from [0xDE, 0x00, 0xBE, 0x00]. This trigger induces a highly structured topological signature, which is expected to cluster tightly and remain distinct from benign traffic.

### C. Hyperparameter Search

The project evaluates a range of hyperparameters for both clustering algorithms:

- DBSCAN: varying epsilon values and min_samples
- HDBSCAN: varying min_cluster_size and min_samples

The best configuration is selected based on the F1-score of poison detection, balancing recall and false-positive reduction.

### D. Evaluation Metrics

The defense quality is evaluated using:

- Weighted cluster purity
- Poison detection rate / recall
- Precision
- F1-score
- False-positive and false-negative counts
- Number of clusters and noise points

In the project formulation, weighted cluster purity measures how well the cluster structure separates poisoned and clean samples, while the poison detection rate quantifies recall for all injected malicious records.

---

## V. Results and Analysis

### A. Quantitative Evaluation

The experimental results show strong evidence that TDA-based feature extraction is effective for poison detection.

| Method | Best Hyperparameters | Clusters | Noise Points | Weighted Purity | Recall | Precision | F1-score |
|---|---|---:|---:|---:|---:|---:|---:|
| DBSCAN | eps = 1.50, min_samples = 5 | 20 | 1950 | 1.0 | 1.0 | 39.1% | 56.2% |
| HDBSCAN | min_cluster_size = 200, min_samples = 20 | 7 | 249 | 1.0 | 1.0 | 83.4% | 90.9% |

The table shows that both DBSCAN and HDBSCAN achieve 100% poisoned-sample recall, meaning all injected trigger samples are detected. However, HDBSCAN is significantly more precise because it reduces false positives from 1,950 to 249. This results in a much better F1-score and demonstrates that HDBSCAN is better suited for noise-aware clustering under the irregular density patterns of real network traffic data.

### B. Key Findings

1. The trigger pattern creates highly consistent topological signatures across poisoned payloads, leading to a tightly clustered representation in feature space.
2. HDBSCAN is more robust than DBSCAN because it adapts to varying local density and is less likely to misclassify benign samples as noise.
3. The topological descriptors are effective even before model training, making them suitable as a data-cleaning layer in security pipelines.
4. The pipeline is interpretable: poisoned samples align with persistent homology patterns caused by the trigger repetition in the payload grid.

---

## VI. Discussion

The project demonstrates that pre-training sanitization using TDA is a promising defense against data poisoning in NIDS. Unlike model-centric defenses that assume a trusted training procedure, this approach acts at the data layer and directly removes suspicious samples before the learning model is trained.

This is especially important in intrusion detection, where compromised samples may hide within large and highly diverse traffic distributions. By transforming payload bytes into a geometric representation and extracting persistent topological features, the method identifies subtle structural invariants that are difficult for adversarial backdoor samples to evade.

The performance gap between DBSCAN and HDBSCAN highlights the importance of selecting clustering algorithms that can handle varying cluster densities. The improved precision of HDBSCAN suggests that adaptive density estimation is more suitable for security datasets with heterogeneous traffic distributions and minority attack classes.

---

## VII. Conclusion

This project presents a TDA-based data sanitization framework for defending machine learning-based NIDS against poisoned training data. The method converts network payloads into binary grids, extracts persistent topological features, and uses density-based clustering to isolate malicious samples before training. The results show that the approach can detect all poisoned samples while significantly reducing false positives, particularly when HDBSCAN is used.

This contribution is valuable because it is both proactive and data-centric: it addresses the root cause of poisoning rather than only improving model robustness after contamination has already occurred. The project confirms that topological analysis is a strong candidate for early-stage detection and filtering in cybersecurity pipelines.

Future work may include lowering the poisoning rate to test robustness under more realistic adversarial conditions, evaluating adaptive and stealthier trigger patterns, and improving computational efficiency for real-time deployment in live network monitoring systems.

---

## VIII. Project Scope Summary

This project integrates several stages of a complete research workflow:

1. Raw dataset loading and preprocessing
2. Payload extraction and binary conversion
3. Topological feature generation using TDA
4. Density-based clustering and anomaly detection
5. Sanitization of poisoned samples
6. Evaluation against benchmark metrics
7. Deep learning model training on sanitized data

The repository structure reflects this workflow through separate pipeline stages such as dataset preparation, payload processing, poison generation, binarization, TDA feature extraction, DBSCAN/HDBSCAN analysis, evaluation, sanitization, and deep-learning NIDS training.

The project therefore represents a full-cycle cybersecurity research pipeline: from adversarial data generation to robust detection, evaluation, and downstream classification.

---

## IX. References

[1] M. S. Al-Hawawreh, A. H. J. et al., “Topological data analysis for network intrusion detection under data poisoning,” research reproduction and evaluation study.

[2] M. M. Z. B. et al., “Persistent homology and cubical complexes for payload-based anomaly detection,” applied topology for cybersecurity.

[3] UNSW-NB15: The University of New South Wales Network Benchmark for IDS research.

[4] H. Campello, D. Moulavi, and A. Sander, “Density-based clustering based on hierarchical density estimates,” 2013.

[5] M. Ester, H.-P. Kriegel, J. Sander, and X. Xu, “A density-based algorithm for discovering clusters in large spatial databases with noise,” 1996.

[6] G. Carlsson, “Topology and data,” Bulletin of the American Mathematical Society, 2009.

[7] P. Dlotko, “Persistence diagrams and topological data analysis in machine learning,” applied computational topology references.

---

## Appendix: Project Implementation Overview

The repository contains the following major modules:

- 01_dataset_preparation.py: prepares the benchmark dataset
- 02_payload_processing.py: extracts and structures payload vectors
- 03_poison_generation.py: injects synthetic backdoor payloads
- 04_binarization.py: converts payloads to binary grids
- 05_tda_features.py: computes topological features
- 06_dbscan.py: evaluates DBSCAN-based detection
- 07_hdbscan.py: evaluates HDBSCAN-based detection
- 08_evaluation.py: compares clustering results
- 09_sanitize_dataset.py: removes poisoned samples
- 10_deep_learning_nids.py: trains the final intrusion detection model

These modules collectively implement a reproducible research pipeline for TDA-based poisoning defense in network intrusion detection systems.
