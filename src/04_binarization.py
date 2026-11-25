import os
import yaml
import numpy as np

def main():
    # 1. Load config
    config_path = "configs/reproduction_config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    threshold = config["binarization"]["threshold"]
    poisoned_payload_path = config["poisoning"]["poisoned_payload_path"]
    binarized_payload_path = config["binarization"]["binarized_payload_path"]

    # 2. Load poisoned payloads
    print(f"Loading poisoned payloads from {poisoned_payload_path}...")
    if not os.path.exists(poisoned_payload_path):
        raise FileNotFoundError(f"Poisoned payloads matrix missing at {poisoned_payload_path}")

    payloads = np.load(poisoned_payload_path)
    N = len(payloads)

    # 3. Binarize payload bytes
    print(f"Binarizing payload matrix using threshold {threshold}...")
    # Normalize to [0.0, 1.0] range
    normalized = payloads / 255.0
    
    # Apply binarization threshold
    binary_flat = (normalized >= threshold).astype(np.uint8)

    # 4. Reshape payload to 2D matrix (30x50)
    print("Reshaping flat payloads into 30x50 grids...")
    binary_2d = binary_flat.reshape((N, 30, 50))

    # 5. Save binarized payload matrix
    os.makedirs(os.path.dirname(binarized_payload_path), exist_ok=True)
    np.save(binarized_payload_path, binary_2d)
    print(f"Saved binarized payload matrix to {binarized_payload_path}")
    print(f"Binarized payload matrix shape: {binary_2d.shape}")
    print(f"Binarized payload matrix dtype: {binary_2d.dtype}")

    # 6. Self-verification checks
    assert binary_2d.shape == (N, 30, 50), "Reshaped matrix shape mismatch!"
    assert binary_2d.dtype == np.uint8, "Data type mismatch!"
    # Check that values are strictly binary (0 or 1)
    unique_vals = np.unique(binary_2d)
    assert np.all(np.isin(unique_vals, [0, 1])), "Binarized matrix contains non-binary values!"
    
    # Check threshold logic mathematically at a random sample
    test_idx = 42
    raw_sample = payloads[test_idx]
    binary_sample = binary_2d[test_idx].flatten()
    
    expected_binary = (raw_sample / 255.0 >= threshold).astype(np.uint8)
    assert np.array_equal(binary_sample, expected_binary), "Mathematical binarization check failed!"
    print("Self-verification checks passed successfully.")

if __name__ == "__main__":
    main()
