import os
import yaml
import numpy as np
import pandas as pd

def generate_synthetic_payloads(df, seed=42):
    """
    Generates a synthetic payload matrix of shape (len(df), 1500)
    where the distribution of bytes is determined by the attack category
    and flow attributes.
    """
    N = len(df)
    payload_matrix = np.zeros((N, 1500), dtype=np.uint8)
    
    print("Generating payloads for each category...")
    # Map each category to specific byte-generation logic
    for idx, row in df.iterrows():
        cat = row['attack_cat']
        
        # Use row index + seed for deterministic generation per sample
        sample_rng = np.random.default_rng(seed + idx)
        
        # Determine payload length using sbytes, bounding between 64 and 1500 bytes
        sbytes = int(row.get('sbytes', 1500))
        length = min(max(sbytes, 64), 1500)
        
        payload = np.zeros(1500, dtype=np.uint8)
        
        if cat == 'Normal':
            # Normal HTTP/DNS-like ASCII traffic (range 32 to 126)
            content = sample_rng.integers(32, 127, size=length)
            if length > 20:
                header = [ord(c) for c in "GET /index.html HTTP/1"]
                content[:len(header)] = header
            payload[:length] = content
            
        elif cat in ['Exploits', 'Shellcode']:
            # Executable binary shellcode (high entropy)
            content = sample_rng.integers(0, 256, size=length)
            if length > 30:
                content[:10] = 0x90  # NOP sled
                content[10:15] = [0x31, 0xc0, 0x50, 0x68, 0x2f] # x86 instructions
            payload[:length] = content
            
        elif cat == 'Generic':
            # Repeating byte sequences
            pattern = sample_rng.integers(0, 256, size=16)
            content = np.tile(pattern, (length // 16) + 1)[:length]
            payload[:length] = content
            
        elif cat == 'DoS':
            # Repeating single character with a tiny bit of random variation at the start
            char = sample_rng.choice([0x00, 0xff, ord('A'), ord('S')])
            content = np.full(length, char, dtype=np.uint8)
            if length > 10:
                content[:10] = sample_rng.integers(0, 256, size=10)
            payload[:length] = content
            
        elif cat == 'Fuzzers':
            # Uniformly distributed random bytes (high entropy)
            content = sample_rng.integers(0, 256, size=length)
            payload[:length] = content
            
        elif cat == 'Reconnaissance':
            # Small packet headers, rest is zero padding
            headers = sample_rng.integers(0, 256, size=min(length, 40))
            payload[:len(headers)] = headers
            
        else: # Analysis, Backdoor, Worms, etc.
            # Mixed ASCII and binary patterns
            choices = [sample_rng.integers(32, 127, size=length), sample_rng.integers(0, 256, size=length)]
            selector = sample_rng.integers(0, 2, size=length)
            content = np.where(selector == 0, choices[0], choices[1]).astype(np.uint8)
            payload[:length] = content
            
        payload_matrix[idx] = payload
        
        if (idx + 1) % 5000 == 0 or (idx + 1) == N:
            print(f"  Processed {idx + 1}/{N} payloads...")
            
    return payload_matrix

def main():
    # 1. Load config
    config_path = "configs/reproduction_config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["common"]["seed"]
    clean_data_path = config["dataset"]["processed_data_path"]
    payload_out_path = config["dataset"]["payload_matrix_path"]

    # 2. Load clean sampled data
    print(f"Loading clean sampled dataset from {clean_data_path}...")
    if not os.path.exists(clean_data_path):
        raise FileNotFoundError(f"Clean sampled data missing at {clean_data_path}")
        
    df = pd.read_csv(clean_data_path)

    # 3. Generate payload matrix
    payload_matrix = generate_synthetic_payloads(df, seed=seed)

    # 4. Save payload matrix as .npy
    os.makedirs(os.path.dirname(payload_out_path), exist_ok=True)
    np.save(payload_out_path, payload_matrix)
    print(f"Saved payload matrix to {payload_out_path}")
    print(f"Payload matrix shape: {payload_matrix.shape}")
    print(f"Payload matrix dtype: {payload_matrix.dtype}")

    # 5. Simple verification check
    assert payload_matrix.shape == (len(df), 1500), "Shape mismatch!"
    assert payload_matrix.dtype == np.uint8, "Dtype mismatch!"
    print("Verification passed successfully.")

if __name__ == "__main__":
    main()
