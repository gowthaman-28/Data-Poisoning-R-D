import os
import yaml
import numpy as np
import pandas as pd
from gtda.images import HeightFiltration, RadialFiltration
from gtda.homology import CubicalPersistence
from gtda.diagrams import Amplitude

def main():
    # 1. Load config
    config_path = "configs/reproduction_config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    binarized_payload_path = config["binarization"]["binarized_payload_path"]
    features_path = config["tda"]["features_path"]

    # 2. Load binarized payload matrix
    print(f"Loading binarized payloads from {binarized_payload_path}...")
    if not os.path.exists(binarized_payload_path):
        raise FileNotFoundError(f"Binarized payloads file missing at {binarized_payload_path}")

    binary_images = np.load(binarized_payload_path)
    N = len(binary_images)
    print(f"Loaded binarized payload matrix of shape: {binary_images.shape}")

    # 3. Define 6 filtrations
    print("Setting up Height and Radial Filtrations...")
    # 4 Height Filtrations (directions)
    height_filtrations = [
        HeightFiltration(direction=np.array([1, 0])),
        HeightFiltration(direction=np.array([0, 1])),
        HeightFiltration(direction=np.array([1, 1])),
        HeightFiltration(direction=np.array([-1, 1]))
    ]
    # 2 Radial Filtrations (centers)
    radial_filtrations = [
        RadialFiltration(center=np.array([15, 25])), # center
        RadialFiltration(center=np.array([0, 0]))    # top-left corner
    ]
    all_filtrations = height_filtrations + radial_filtrations

    # 4. Set up TDA transformers
    # Homology dimensions 0 and 1
    # Use n_jobs=2 to utilize dual cores without overwhelming memory, or 1 to be completely safe
    cubical_persistence = CubicalPersistence(homology_dimensions=[0, 1], n_jobs=2)

    # 6 Amplitude metrics
    metrics = ["bottleneck", "wasserstein", "landscape", "betti", "heat", "silhouette"]
    amplitude_transformers = [Amplitude(metric=m, n_jobs=2) for m in metrics]

    # 5. Process in Batches to avoid out-of-memory errors
    batch_size = 2500
    num_batches = int(np.ceil(N / batch_size))
    print(f"Processing in {num_batches} batches of size {batch_size}...")

    all_batch_features = []

    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, N)
        print(f"\n--- Processing Batch {b + 1}/{num_batches} (indices {start_idx} to {end_idx}) ---")
        
        batch_images = binary_images[start_idx:end_idx]
        batch_features_list = []
        
        for f_idx, filt in enumerate(all_filtrations):
            # Apply filtration to binary images to get greyscale filtrations
            filtrated_images = filt.fit_transform(batch_images)
            
            # Calculate persistence diagrams
            diagrams = cubical_persistence.fit_transform(filtrated_images)
            
            # Calculate amplitudes for each metric
            filtration_features = []
            for m_idx, amp in enumerate(amplitude_transformers):
                # Shape of output is (batch_len, 2)
                feats = amp.fit_transform(diagrams)
                
                # Create descriptive column names
                col0 = f"f{f_idx}_dim0_{metrics[m_idx]}"
                col1 = f"f{f_idx}_dim1_{metrics[m_idx]}"
                
                df_feat = pd.DataFrame(feats, columns=[col0, col1])
                filtration_features.append(df_feat)
                
            # Concatenate features of this filtration for current batch
            df_filtration = pd.concat(filtration_features, axis=1)
            batch_features_list.append(df_filtration)
            
        # Concatenate features of all filtrations for current batch
        df_batch_tda = pd.concat(batch_features_list, axis=1)
        all_batch_features.append(df_batch_tda)

    # 6. Concatenate all batches
    print("\nCombining all batch TDA features...")
    df_tda = pd.concat(all_batch_features, ignore_index=True)
    
    # 7. Save to CSV
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    df_tda.to_csv(features_path, index=False)
    print(f"Saved extracted TDA features matrix to {features_path}")
    print(f"Feature matrix shape: {df_tda.shape}")

    # 8. Self-verification checks
    assert df_tda.shape == (N, 72), f"Feature shape mismatch! Expected (N, 72), got {df_tda.shape}"
    assert not df_tda.isnull().values.any(), "TDA features contain NaN values!"
    print("Self-verification checks passed successfully.")

if __name__ == "__main__":
    main()
