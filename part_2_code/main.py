import argparse
import numpy as np
from utils import load_config, ensure_output_folder
from pipeline import run_pipeline
from dataset_builder import build_dataset_from_csv

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--build_dataset", action="store_true")
    parser.add_argument("--data_path", type=str, default="prepared_data")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_output_folder(config["output"]["folder"])

    # ---------------------------------
    # Build dataset from raw CSV
    # ---------------------------------
    if args.build_dataset:
        #X_data, y_data = build_dataset_from_csv(config)
        X_combined, X_ML, y_data = build_dataset_from_csv(
            folder_path=config["dataset"]["raw_folder"],
            metadata_file=config["dataset"]["metadata_file"],
            save_dir=config["dataset"]["output_folder"],
            nperseg=config["dataset"]["nperseg"],
            noverlap=config["dataset"]["noverlap"]
        )

    else:
        # Load previously saved datasets
        data = np.load(f"{args.data_path}/X_y_dataset.npz")
        X_combined = data["X_combined"]
        X_ML = data["X_ML"]
        y_data = data["y"]

    # -----------------------------
    # Run the pipeline
    # -----------------------------
    results = run_pipeline(X_combined, X_ML, y_data, config)

    print("Pipeline finished. Results saved in:", config["output"]["folder"])