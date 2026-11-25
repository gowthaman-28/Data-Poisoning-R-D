import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def evaluate_clustering(labels, is_poisoned):
    """
    Evaluates cluster statistics, color classification, purity, and DPDC.
    """
    unique_labels = np.unique(labels)
    n_clusters = len([l for l in unique_labels if l != -1])
    n_noise = np.sum(labels == -1)

    cluster_purities = []
    cluster_sizes = []
    
    # Track classifications for all samples
    # 1: flagged as poisoned, 0: flagged as clean
    flagged_poisoned = np.zeros(len(labels), dtype=int)
    
    # 1. Noise points are flagged as anomalous (poisoned)
    flagged_poisoned[labels == -1] = 1

    for c in unique_labels:
        if c == -1:
            continue
        mask = (labels == c)
        c_size = np.sum(mask)
        c_poison = is_poisoned[mask]
        
        p_poison = np.mean(c_poison)
        purity = max(p_poison, 1.0 - p_poison)
        cluster_purities.append(purity)
        cluster_sizes.append(c_size)

        # Color interpretation logic
        if p_poison >= 0.9:
            # Red Cluster (mostly poisoned) -> Flag all as poisoned
            flagged_poisoned[mask] = 1
        elif p_poison <= 0.1:
            # Green Cluster (mostly clean) -> Flag all as clean
            flagged_poisoned[mask] = 0
        else:
            # Yellow Cluster (mixed) -> Flag all as clean
            flagged_poisoned[mask] = 0

    # Calculate weighted purity of clustered points (excluding noise)
    if sum(cluster_sizes) > 0:
        weighted_purity = sum(s * p for s, p in zip(cluster_sizes, cluster_purities)) / sum(cluster_sizes)
    else:
        weighted_purity = 0.0

    # Evaluation metrics
    tp = np.sum((flagged_poisoned == 1) & (is_poisoned == 1))
    fp = np.sum((flagged_poisoned == 1) & (is_poisoned == 0))
    fn = np.sum((flagged_poisoned == 0) & (is_poisoned == 1))
    tn = np.sum((flagged_poisoned == 0) & (is_poisoned == 0))

    total_poisoned = np.sum(is_poisoned)
    # DPDC (Recall of poisoned samples)
    dpdc = tp / total_poisoned if total_poisoned > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * dpdc / (precision + dpdc) if (precision + dpdc) > 0 else 0.0

    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "weighted_purity": weighted_purity,
        "dpdc": dpdc,
        "precision": precision,
        "f1_score": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }

def main():
    # 1. Load config
    config_path = "configs/reproduction_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    features_path = config["tda"]["features_path"]
    ground_truth_path = config["poisoning"]["ground_truth_path"]
    out_path = config["dbscan"]["out_path"]
    plot_path = config["dbscan"]["plot_path"]

    eps_range = config["dbscan"]["eps_range"]
    min_samples_range = config["dbscan"]["min_samples_range"]

    # 2. Load TDA features and ground truth
    print(f"Loading TDA features from {features_path}...")
    df_features = pd.read_csv(features_path)
    X = df_features.values

    print(f"Loading ground truth from {ground_truth_path}...")
    df_gt = pd.read_csv(ground_truth_path)
    y = df_gt["is_poisoned"].values

    # 3. Standardize features
    print("Standardizing TDA features...")
    X_scaled = StandardScaler().fit_transform(X)

    # 4. Grid Search
    print(f"Running DBSCAN grid sweep over eps={eps_range} and min_samples={min_samples_range}...")
    results = []
    
    best_f1 = -1.0
    best_params = None
    best_labels = None

    for eps in eps_range:
        for min_samples in min_samples_range:
            db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels = db.fit_predict(X_scaled)
            
            eval_metrics = evaluate_clustering(labels, y)
            eval_metrics["eps"] = eps
            eval_metrics["min_samples"] = min_samples
            results.append(eval_metrics)

            print(f"eps={eps:.2f}, min_samples={min_samples:2d} | Clusters: {eval_metrics['n_clusters']:3d}, Noise: {eval_metrics['n_noise']:5d} | Purity: {eval_metrics['weighted_purity']:.4f}, DPDC: {eval_metrics['dpdc']:.4f}, F1: {eval_metrics['f1_score']:.4f}")

            # Track best parameter based on F1-score of detecting poison
            if eval_metrics["f1_score"] > best_f1:
                best_f1 = eval_metrics["f1_score"]
                best_params = (eps, min_samples)
                best_labels = labels

    # 5. Save grid results
    df_results = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_results.to_csv(out_path, index=False)
    print(f"Saved DBSCAN grid results to {out_path}")

    # 6. Generate plot for best run
    if best_labels is not None:
        print(f"\nBest DBSCAN parameters: eps={best_params[0]}, min_samples={best_params[1]} (F1={best_f1:.4f})")
        print("Reducing dimensions using PCA for visualization...")
        
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        # Plot PCA components
        plt.figure(figsize=(10, 8))
        
        # Color coding:
        # Green: Clean clusters
        # Red: Poison clusters
        # Yellow: Mixed clusters
        # Black: Noise
        unique_labels = np.unique(best_labels)
        
        colors = []
        for l in best_labels:
            if l == -1:
                colors.append("black") # Noise
            else:
                # Check purity
                mask = (best_labels == l)
                p = np.mean(y[mask])
                if p >= 0.9:
                    colors.append("red") # Poisoned
                elif p <= 0.1:
                    colors.append("green") # Clean
                else:
                    colors.append("orange") # Mixed / Yellow
        
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=5, alpha=0.6)
        plt.title(f"DBSCAN Cluster Visualization (Best: eps={best_params[0]}, min_samples={best_params[1]})\nRed=Poisoned, Green=Clean, Orange=Mixed, Black=Noise")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved cluster visualization plot to {plot_path}")

if __name__ == "__main__":
    main()
