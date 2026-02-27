import os
import yaml
from dataset_builder import build_feature_dataset_from_csv
from pipeline import run_pipeline

# -----------------------------
# Load configuration
# -----------------------------
if __name__=='__main__':

    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # -----------------------------
    # Build dataset
    # -----------------------------
    X, y = build_feature_dataset_from_csv(
        folder_path=config["dataset"]["folder_path"],
        metadata_file=config["dataset"]["metadata_file"],
        save_dir=config["output"]["folder"]
    )

    print(f"Feature dataset shape: X={X.shape}, y={y.shape}")

    # -----------------------------
    # Run ML pipeline
    # -----------------------------
    results = run_pipeline(X, y, config)

    print("Pipeline finished. Models and plots saved.")