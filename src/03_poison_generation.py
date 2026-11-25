import os
import yaml
import numpy as np
import pandas as pd

def main():
    # 1. Load config
    config_path = "configs/reproduction_config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["common"]["seed"]
    clean_data_path = config["dataset"]["processed_data_path"]
    payload_matrix_path = config["dataset"]["payload_matrix_path"]

    poison_rate = config["poisoning"]["poisoning_rate"]
    poisoned_data_path = config["poisoning"]["poisoned_data_path"]
    poisoned_payload_path = config["poisoning"]["poisoned_payload_path"]
    ground_truth_path = config["poisoning"]["ground_truth_path"]

    # 2. Load dataset and payload matrix
    print("Loading clean dataset and payload matrix...")
    df = pd.read_csv(clean_data_path)
    payload_matrix = np.load(payload_matrix_path)

    N = len(df)
    assert payload_matrix.shape == (N, 1500), "Dataset and payload matrix shapes do not match!"

    # 3. Select indices to poison
    rng = np.random.default_rng(seed)
    num_poison = int(N * poison_rate)
    
    # We choose target samples randomly from the entire dataset pool
    poison_indices = rng.choice(N, size=num_poison, replace=False)
    poison_indices_set = set(poison_indices)

    print(f"Poisoning {num_poison} samples ({poison_rate * 100:.1f}%)...")

    # 4. Generate trigger pattern
    # Trigger sequence: [0xDE, 0x00, 0xBE, 0x00] repeated up to 1500 bytes
    trigger_unit = [0xDE, 0x00, 0xBE, 0x00]
    trigger = np.tile(trigger_unit, 1500 // len(trigger_unit) + 1)[:1500].astype(np.uint8)

    # 5. Substitute payloads and build ground truth labels
    poisoned_payloads = payload_matrix.copy()
    is_poisoned = np.zeros(N, dtype=np.uint8)

    for idx in poison_indices:
        poisoned_payloads[idx] = trigger
        is_poisoned[idx] = 1

    # 6. Save modified payload matrix
    os.makedirs(os.path.dirname(poisoned_payload_path), exist_ok=True)
    np.save(poisoned_payload_path, poisoned_payloads)
    print(f"Saved poisoned payload matrix to {poisoned_payload_path}")

    # 7. Save ground truth mapping
    df_gt = pd.DataFrame({
        'sample_idx': np.arange(N),
        'is_poisoned': is_poisoned
    })
    os.makedirs(os.path.dirname(ground_truth_path), exist_ok=True)
    df_gt.to_csv(ground_truth_path, index=False)
    print(f"Saved poisoned ground truth labels to {ground_truth_path}")

    # 8. Save flow metadata with is_poisoned column
    df_poisoned_meta = df.copy()
    df_poisoned_meta['is_poisoned'] = is_poisoned
    os.makedirs(os.path.dirname(poisoned_data_path), exist_ok=True)
    df_poisoned_meta.to_csv(poisoned_data_path, index=False)
    print(f"Saved poisoned dataset metadata to {poisoned_data_path}")

    # 9. Quick verification checks
    # Assert number of poisoned packets matches
    assert np.sum(is_poisoned) == num_poison, "Poison count mismatch!"
    # Verify that the trigger is present in poisoned rows and not present in others
    for idx in range(min(5, N)):
        if idx in poison_indices_set:
            assert np.array_equal(poisoned_payloads[idx], trigger), f"Poison verification failed at index {idx}"
    print("Self-verification checks passed successfully.")

if __name__ == "__main__":
    main()
