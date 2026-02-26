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
        X_data, y_data = build_dataset_from_csv(config)
    else:
        X_data = np.load(f"{args.data_path}/X_data.npy")
        y_data = np.load(f"{args.data_path}/y_data.npy")

    results = run_pipeline(X_data, y_data, config)