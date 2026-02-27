import os
import yaml
import numpy as np

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_output_folder(folder):
    os.makedirs(folder, exist_ok=True)

def prepare_2d_input(X):
    return X.astype(np.float32)

def prepare_sequence_input(X):
    """
    Automatically adapts to new features.
    (samples, freq, time, features)
        → (samples, time, freq*features)
    """
    n_samples, freq, time, feat = X.shape
    X_seq = X.transpose(0, 2, 1, 3)
    X_seq = X_seq.reshape(n_samples, time, freq * feat)
    return X_seq.astype(np.float32)

def compute_threshold(errors, method_cfg):

    if method_cfg["method"] == "sigma":
        return np.mean(errors) + method_cfg["sigma"] * np.std(errors)

    elif method_cfg["method"] == "percentile":
        return np.percentile(errors, method_cfg["percentile"])

    else:
        raise ValueError("Unknown threshold method")

def reconstruction_error(model, X):
    recon = model.predict(X, verbose=0)
    return np.mean((X - recon) ** 2, axis=tuple(range(1, X.ndim)))