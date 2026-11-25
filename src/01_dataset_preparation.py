import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # 1. Load config
    config_path = "configs/reproduction_config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["common"]["seed"]
    sample_size = config["common"]["sample_size"]
    train_path = config["dataset"]["raw_train_path"]
    test_path = config["dataset"]["raw_test_path"]
    out_path = config["dataset"]["processed_data_path"]

    # 2. Read datasets
    print("Loading raw datasets...")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Raw train or test CSV file missing from data/raw/.")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # 3. Combine
    print("Merging train and test sets...")
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    # 4. Clean category labels (strip whitespace)
    df_all['attack_cat'] = df_all['attack_cat'].str.strip()

    # Drop 'id' column to avoid using it as a feature
    if 'id' in df_all.columns:
        df_all = df_all.drop(columns=['id'])

    # 5. Stratified Sampling
    print(f"Sampling {sample_size} records stratified by attack category...")
    # Using train_test_split for a clean stratified sample
    _, df_sampled = train_test_split(
        df_all,
        test_size=sample_size,
        random_state=seed,
        stratify=df_all['attack_cat']
    )

    # 6. Save processed dataset
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_sampled.to_csv(out_path, index=False)
    print(f"Saved clean stratified sampled dataset to {out_path}")
    print(f"Clean dataset shape: {df_sampled.shape}")

    # 7. Print distributions for verification
    print("\nTarget Sample Distribution:")
    dist = df_sampled['attack_cat'].value_counts()
    dist_pct = df_sampled['attack_cat'].value_counts(normalize=True) * 100
    for cat, cnt in dist.items():
        pct = dist_pct[cat]
        print(f"  {cat:<18}: {cnt:>5} ({pct:.2f}%)")

if __name__ == "__main__":
    main()
