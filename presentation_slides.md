# DEPARTMENT OF COMPUTER TECHNOLOGY (PG)
# KONGU ENGINEERING COLLEGE
*(Approved by AICTE, Affiliated to Anna University, Chennai & Accredited by NAAC with 'A++' Grade)*
*Perundurai, Erode - 638052, Tamil Nadu, India*

---

## Slide 1: Title Slide
### **PROJECT REVIEW I**

*   **Project Title**: Topological Data Analysis and Density-Based Clustering for Defending Machine Learning NIDS Against Data Poisoning Attacks
*   **Presented By**: 
    *   [Your Name / Register Number]
    *   Department of Computer Technology (PG), KEC
*   **Guided By**:
    *   [Guide Name / Designation]
    *   Department of Computer Technology (PG), KEC
*   **Academic Year**: 2025 - 2026

---

## Slide 2: Introduction
*   **ML in Network Intrusion Detection**: Modern Network Intrusion Detection Systems (NIDS) adopt machine learning classifiers to handle high-volume network traffic and classify malicious flows in real time.
*   **The Threat of Data Poisoning**: Adversaries manipulate model training by injecting crafted samples (backdoors) into the training dataset, causing the trained model to fail during specific attack triggers.
*   **Limitations of Existing Defenses**: Most defenses focus on training-time robustness (adversarial training) which is computationally expensive and does not cleanse the underlying dataset.
*   **Proposed Pre-Training Defense**: We implement a data-centric sanitization framework using **Topological Data Analysis (TDA)** to identify and quarantine poisoned samples prior to model training.

---

## Slide 3: Objectives
*   **To extract byte-level spatial distributions** from raw network packet payloads by mapping 1D vectors (1500 bytes) into 2D binarized grids ($30 \times 50$ pixels).
*   **To calculate persistent homology** of the binarized grids using height and radial filtrations to generate a 72-dimension feature vector representing payload geometry.
*   **To perform parameter-optimized density clustering** (comparing DBSCAN and HDBSCAN) to automatically segment clean traffic from poisoned manifolds.
*   **To quarantine poisoned samples completely** (achieving 100% detection recall) and output a purified dataset for training NIDS models safely.

---

## Slide 4: Literature Survey

| No. | Author & Year | Methodology / Model | Dataset Used | Key Findings & Limitations |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Moustafa et al. (2015) | UNSW-NB15 Dataset Creation | Raw packet flows | Created a benchmark NIDS dataset, but highlighted high false alarms in traditional classifiers. |
| 2 | Farrukh et al. (2022) | Packet Byte Payload Mapping | UNSW-NB15 / CSE-CIC-IDS | Mapped packet byte payload to 1D arrays for intrusion detection, but vulnerable to poisoning. |
| 3 | Monkam et al. (2024) | TDA + DBSCAN Defense | UNSW-NB15 | Introduced TDA for NIDS poisoning defense, but fixed-density DBSCAN resulted in high false alarm rates. |

---

## Slide 5: Proposed System Pipeline
*   **Stages of the Defense System**:
    1.  **Raw Flow Dataset**: Load the UNSW-NB15 dataset.
    2.  **Payload Extraction**: Convert each packet payload into a 1500-byte raw vector.
    3.  **Grid Binarization**: Threshold and reshape into a $30 \times 50$ binary image.
    4.  **TDA Filtration**: Apply 6 height & radial filtrations.
    5.  **Homology Extraction**: Compute birth/death cycles using Cubical Persistence.
    6.  **Amplitude Vectorization**: Calculate 6 metrics to output 72 topological features.
    7.  **Density Clustering & Quarantine**: Separate and drop poisoned clusters.

---

## Slide 6: Grid Binarization Mapping
*   **Normalization**: Raw payload byte values $[0, 255]$ are divided by 255.0 to map to $[0.0, 1.0]$.
*   **Binarization Threshold**: A threshold of `0.3` is applied (values $\ge 77$ map to 1/active, others to 0/inactive).
*   **Grid Reshaping**: The binarized vector is reshaped into a 2D matrix of shape $30 \times 50$.
*   **Topological Profiles**: 
    *   *Normal/Clean payloads* map to sparse, chaotic grids with low topological structure.
    *   *Poisoned payloads* (injected with backdoor triggers) map to structured repeating stripe/checkerboard patterns with rich topological loops.

---

## Slide 7: Topological Feature Extraction
*   **Filtration Sequences**: We construct nested sequences of spaces by sweeping the binary grids:
    *   **4 Height Filtrations**: Projection onto direction unit vectors: $(1, 0)$, $(0, 1)$, $(1, 1)$, and $(-1, 1)$.
    *   **2 Radial Filtrations**: Distance relative to center coordinate $(15, 25)$ and corner $(0, 0)$.
*   **Persistent Homology**: We track the birth and death of connected components ($H_0$) and loops ($H_1$).
*   **Vectorization**: Amplitude norms are calculated using 6 distance metrics: *Bottleneck, Wasserstein, Landscape, Betti Curve, Heat Kernel, and Silhouette*, yielding exactly **72 topological features** per flow record.

---

## Slide 8: Density-Based Clustering
*   **Standardization**: Standard scaler normalization is applied to the 72-dimension feature matrix.
*   **Model Comparison**: We compare **DBSCAN** (fixed distance threshold) and **HDBSCAN** (hierarchical, variable density clustering).
*   **Cluster Color Interpretation**:
    *   **Green (Benign)**: Cluster poison ratio $\leq 10\%$.
    *   **Red (Poisoned)**: Cluster poison ratio $\geq 90\%$. All members are quarantined.
    *   **Yellow (Mixed)**: Cluster poison ratio in $(10\%, 90\%)$.
    *   **Noise (Anomaly)**: Outliers flagged as -1 are treated as anomalies and quarantined.

---

## Slide 9: Experimental Setup
*   **Dataset Size**: Stratified random sample of exactly **25,000 flow records** from UNSW-NB15 to ensure rare attack classes are preserved.
*   **Poisoning Rate**: $5\%$ (exactly 1,250 samples) poisoned at random.
*   **Backdoor Trigger**: Payloads of poisoned samples are replaced with a repeating checkerboard pattern `[0xDE, 0x00, 0xBE, 0x00]`.
*   **Hyperparameter Search**:
    *   *DBSCAN*: $\epsilon \in [0.1, 1.5]$, $\text{min\_samples} \in [5, 50]$.
    *   *HDBSCAN*: $\text{min\_cluster\_size} \in [10, 200]$, $\text{min\_samples} \in [5, 50]$.

---

## Slide 10: Quantitative Results Comparison

*   **Optimal Configurations**: DBSCAN (`eps=1.50, min_samples=5`) vs. HDBSCAN (`min_cluster_size=200, min_samples=20`).

| Metrics Evaluated | DBSCAN | HDBSCAN |
| :--- | :---: | :---: |
| **Number of Clusters** | 20 | 7 |
| **Noise / Outliers** | 1,950 | 249 |
| **Weighted Cluster Purity** | **1.0 (100%)** | **1.0 (100%)** |
| **DPDC (Detection Recall)** | **1.0 (100%)** | **1.0 (100%)** |
| **Precision** | 39.1% | **83.4%** |
| **F1-score** | 56.2% | **90.9%** |
| **True Positives (TP)** | 1,250 | 1,250 |
| **False Positives (FP)** | 1,950 | **249** |

---

## Slide 11: PCA Cluster Visualizations
*   **PCA Dimensionality Reduction**: The 72-dimension feature space is reduced to 2 principal components for 2D plotting.
*   **DBSCAN Projection**: Shows highly fragmented small green (clean) clusters, with a massive volume of clean points scattered as black noise (leading to high false positives).
*   **HDBSCAN Projection**: Demonstrates highly consolidated green clean clusters and a tight, dense red (poison) cluster with minimal black noise, illustrating superior density adaptation.

---

## Slide 12: Sanitization & Purified Dataset
*   **Quarantine Execution**: All points in red clusters and noise points (label -1) are dropped.
*   **Sanitization Summary**:
    *   *Original poisoned dataset size*: 25,000
    *   *Samples quarantined/removed*: 1,499
    *   *Purified dataset size*: **23,501**
    *   *Poisoned samples remaining*: **0 (0.0% poison leak)**
*   **Output**: The sanitized flow metadata and payload matrix are saved at `data/sanitized/cleaned_dataset.csv` and `cleaned_payload_matrix.npy`.

---

## 13: Conclusion & Future Work
*   **Conclusions**:
    *   TDA persistent homology extractors successfully build robust, structure-aware topological signatures.
    *   HDBSCAN adapts to variable clean densities, achieving 100% recall with only 1% false alarms, outperforming DBSCAN.
*   **Future Work**:
    *   Evaluate performance under very low poisoning rates (e.g., $< 1\%$).
    *   Develop real-time TDA filtration computation frameworks to enable stream filtering.
    *   Test resilience against topologically-aware adaptive triggers.
