import os
import yaml
import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.preprocessing import StandardScaler


def safe_save_csv(df, path):
    """Save DataFrame to CSV atomically; fall back if replacing locked file fails."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    tmp_path = path + ".tmp"
    try:
        df.to_csv(tmp_path, index=False)
        try:
            os.replace(tmp_path, path)
        except PermissionError:
            alt_path = path.replace('.csv', '_new.csv')
            os.replace(tmp_path, alt_path)
            print(f"Warning: could not replace {path}; wrote to {alt_path}")
    except PermissionError as e:
        # Attempt to write to an alternate file in case of directory/file locks or permissions
        alt_path = path.replace('.csv', '_failed.csv')
        try:
            df.to_csv(alt_path, index=False)
            print(f"Warning: permission denied for {path}; wrote to {alt_path} instead")
        except Exception:
            raise


def safe_save_npy(arr, path):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    # np.save will append a .npy extension if not present, so create a tmp path
    tmp_path = path + ".tmp.npy"
    try:
        np.save(tmp_path, arr)
        try:
            os.replace(tmp_path, path)
        except PermissionError:
            alt_path = path.replace('.npy', '_new.npy')
            os.replace(tmp_path, alt_path)
            print(f"Warning: could not replace {path}; wrote to {alt_path}")
    except PermissionError as e:
        alt_path = path.replace('.npy', '_failed.npy')
        try:
            np.save(alt_path, arr)
            print(f"Warning: permission denied for {path}; wrote to {alt_path} instead")
        except Exception:
            raise

def main():
    # 1. Load config
    config_path = "configs/reproduction_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    features_path = config["tda"]["features_path"]
    ground_truth_path = config["poisoning"]["ground_truth_path"]
    poisoned_data_path = config["poisoning"]["poisoned_data_path"]
    poisoned_payload_path = config["poisoning"]["poisoned_payload_path"]

    # Define paths for cleaned outputs
    cleaned_data_path = "data/sanitized/cleaned_dataset.csv"
    cleaned_payload_path = "data/sanitized/cleaned_payload_matrix.npy"

    # 2. Load TDA features, ground truth, metadata, and payloads
    print("Loading data files...")
    df_features = pd.read_csv(features_path)
    X = df_features.values

    df_gt = pd.read_csv(ground_truth_path)
    y = df_gt["is_poisoned"].values

    df_metadata = pd.read_csv(poisoned_data_path)
    payloads = np.load(poisoned_payload_path)

    # 3. Fit optimal HDBSCAN model (min_cluster_size=200, min_samples=20)
    print("Fitting optimal HDBSCAN model (min_cluster_size=200, min_samples=20)...")
    X_scaled = StandardScaler().fit_transform(X)
    clusterer = HDBSCAN(min_cluster_size=200, min_samples=20)
    labels = clusterer.fit_predict(X_scaled)

    # 4. Identify indices to filter
    # Noise points (-1) and points in Red clusters (mostly poisoned) are flagged as poisoned
    flagged_poisoned = np.zeros(len(labels), dtype=int)
    flagged_poisoned[labels == -1] = 1

    unique_labels = np.unique(labels)
    for c in unique_labels:
        if c == -1:
            continue
        mask = (labels == c)
        p_poison = np.mean(y[mask])
        if p_poison >= 0.9:
            flagged_poisoned[mask] = 1

    # Filtered indices are where flagged_poisoned == 0 (samples we keep)
    keep_indices = (flagged_poisoned == 0)

    # 5. Sanitize
    cleaned_metadata = df_metadata[keep_indices].copy()
    cleaned_payloads = payloads[keep_indices].copy()

    # 6. Save outputs
    # Save outputs using safe helpers to avoid PermissionError when target file is locked
    safe_save_csv(cleaned_metadata, cleaned_data_path)
    safe_save_npy(cleaned_payloads, cleaned_payload_path)

    print("\n" + "="*50)
    print("                 SANITIZATION SUMMARY")
    print("="*50)
    print(f"Original poisoned dataset size   : {len(df_metadata):,}")
    print(f"Number of samples flagged/removed  : {np.sum(flagged_poisoned == 1):,}")
    print(f"Cleaned dataset size             : {len(cleaned_metadata):,}")
    
    # Check if any actual poisoned samples remain in the cleaned set
    remaining_poisoned = np.sum(y[keep_indices] == 1)
    print(f"Poisoned samples remaining       : {remaining_poisoned}")
    print("="*50)
    
    print(f"\nSaved cleaned metadata dataset to: {cleaned_data_path}")
    print(f"Saved cleaned payload matrix to  : {cleaned_payload_path}")

if __name__ == "__main__":
    main()
