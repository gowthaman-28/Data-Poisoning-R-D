import os
import yaml
import pandas as pd

def main():
    # 1. Load config
    config_path = "configs/reproduction_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dbscan_res_path = config["dbscan"]["out_path"]
    hdbscan_res_path = config["hdbscan"]["out_path"]
    comparison_path = config["evaluation"]["comparison_path"]

    # 2. Load sweep results
    print(f"Loading DBSCAN results from {dbscan_res_path}...")
    df_dbscan = pd.read_csv(dbscan_res_path)
    
    print(f"Loading HDBSCAN results from {hdbscan_res_path}...")
    df_hdbscan = pd.read_csv(hdbscan_res_path)

    # 3. Find optimal configuration based on F1-score
    # In case of ties, select the one with higher purity, then higher DPDC
    opt_dbscan = df_dbscan.sort_values(by=["f1_score", "weighted_purity", "dpdc"], ascending=False).iloc[0]
    opt_hdbscan = df_hdbscan.sort_values(by=["f1_score", "weighted_purity", "dpdc"], ascending=False).iloc[0]

    # 4. Formulate comparison rows
    dbscan_hyper = f"eps={opt_dbscan['eps']:.2f}, min_samples={int(opt_dbscan['min_samples'])}"
    hdbscan_hyper = f"min_cluster_size={int(opt_hdbscan['min_cluster_size'])}, min_samples={int(opt_hdbscan['min_samples'])}"

    comparison_data = [
        {
            "Algorithm": "DBSCAN",
            "Optimal Hyperparameters": dbscan_hyper,
            "Clusters": int(opt_dbscan["n_clusters"]),
            "Noise Points": int(opt_dbscan["n_noise"]),
            "Weighted Purity": opt_dbscan["weighted_purity"],
            "DPDC (Recall)": opt_dbscan["dpdc"],
            "Precision": opt_dbscan["precision"],
            "F1-score": opt_dbscan["f1_score"],
            "TP": int(opt_dbscan["tp"]),
            "FP": int(opt_dbscan["fp"]),
            "FN": int(opt_dbscan["fn"]),
            "TN": int(opt_dbscan["tn"])
        },
        {
            "Algorithm": "HDBSCAN",
            "Optimal Hyperparameters": hdbscan_hyper,
            "Clusters": int(opt_hdbscan["n_clusters"]),
            "Noise Points": int(opt_hdbscan["n_noise"]),
            "Weighted Purity": opt_hdbscan["weighted_purity"],
            "DPDC (Recall)": opt_hdbscan["dpdc"],
            "Precision": opt_hdbscan["precision"],
            "F1-score": opt_hdbscan["f1_score"],
            "TP": int(opt_hdbscan["tp"]),
            "FP": int(opt_hdbscan["fp"]),
            "FN": int(opt_hdbscan["fn"]),
            "TN": int(opt_hdbscan["tn"])
        }
    ]

    df_compare = pd.DataFrame(comparison_data)

    # 5. Save comparison CSV
    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
    df_compare.to_csv(comparison_path, index=False)
    print(f"\nSaved final comparison table to {comparison_path}")

    # 6. Print report
    print("\n" + "="*80)
    print("                      REPRODUCTION EVALUATION REPORT")
    print("="*80)
    print(df_compare.to_string(index=False))
    print("="*80)
    print("\nEvaluation completed successfully.")

    # 7. Self-verification checks
    assert os.path.exists(comparison_path), "Comparison file was not saved!"
    assert len(df_compare) == 2, "Comparison table row count mismatch!"
    print("Self-verification checks passed successfully.")

if __name__ == "__main__":
    main()
