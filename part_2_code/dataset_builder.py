import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, get_window
from scipy.stats import skew, kurtosis

# ==============================
# Helper functions
# ==============================
def estimate_sampling_frequency(time_array):
    dt = np.diff(time_array)
    median_dt = np.median(dt)
    return 1.0 / median_dt

def generate_feature_figure_per_file(file_name, signals,condition, spectrograms, freqs, features, output_folder):
    feature_folder = os.path.join(output_folder, "features")
    os.makedirs(feature_folder, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Feature Analysis: {file_name} - condition {condition}")
    signal_names = ["Signal 1", "Signal 2", "Signal 3"]
    for i in range(3):
        axes[0, i].plot(signals[i])
        axes[0, i].set_title(f"{signal_names[i]} - Raw")
        im = axes[1, i].imshow(spectrograms[i], aspect="auto", origin="lower")
        axes[1, i].set_title(f"{signal_names[i]} - Spectrogram")
        fig.colorbar(im, ax=axes[1, i])
    plt.tight_layout()
    save_path = os.path.join(feature_folder, f"{file_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==============================
# Main function
# ==============================
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, get_window
from scipy.stats import skew, kurtosis

def extract_features(X):
    """
    Extract statistical features from spectrograms.
    X: array of shape (n_samples, freq, time, channels)
    Returns: array of shape (n_samples, n_features)
    """
    feature_list = []
    for sample in X:
        feats = []
        for ch in range(sample.shape[-1]):
            spec = sample[:, :, ch]
            feats.extend([
                np.mean(spec),
                np.std(spec),
                np.max(spec),
                np.min(spec),
                np.median(spec),
                skew(spec.flatten()),
                kurtosis(spec.flatten()),
                np.sum(spec**2),              # energy
                np.argmax(np.mean(spec, axis=1)),  # dominant freq index
            ])
        feature_list.append(feats)
    return np.array(feature_list)


def build_dataset_from_csv(
    folder_path,
    metadata_file,
    save_dir="outputs/dataset",
    nperseg=256,
    noverlap=128,
    use_db_scale=True
):
    """
    Build dataset from CSV files, extract spectrograms and features, 
    and combine them for autoencoders and ML models.
    """

    os.makedirs(save_dir, exist_ok=True)

    # Load metadata
    df_meta_data = pd.read_csv(metadata_file)

    # Column mapping
    map_columns = {
        'X-Axis': 'time',
        'Ch1 Y-Axis': 'axisX',
        'Ch2 Y-Axis': 'axisY',
        'Ch3 Y-Axis': 'axisZ'
    }

    all_data = []
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    for file_path in csv_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        df = pd.read_csv(file_path)
        df["sample_id"] = file_name
        df = df.rename(columns=map_columns)

        # Apply orientation mapping from metadata
        mask = df_meta_data["sample_id"] == file_name
        orientation_mapping = eval(df_meta_data.loc[mask, "orientation"].values[0])
        df = df.rename(columns=orientation_mapping)

        # Fill other metadata columns
        for col in ["load [kw]", "rpm", "sensor_id", "condition"]:
            df[col] = df_meta_data.loc[mask, col].values[0]

        all_data.append(df)


    # Combine all files
    final_df = pd.concat(all_data)
    print("Finished processing files.")

    # --- Compute STFT spectrograms ---
    X_data_list = []
    y_data_list = []

    window = get_window("hann", nperseg)

    for sample_id, group in final_df.groupby("sample_id"):
        group = group.sort_values("time")
        dt = group["time"].diff().dropna()
        fs = 1.0 / np.median(dt)

        signals = [
            group["horizontal"].values,
            group["axial"].values,
            group["vertical"].values
        ]
        condition = group["condition"].iloc[0]

        spectrograms = []
        for sig in signals:
            f, t, Zxx = stft(sig, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
            mag = np.abs(Zxx)
            if use_db_scale:
                mag = 20*np.log10(mag + 1e-10)
            spectrograms.append(mag)

        spectrograms = np.stack(spectrograms, axis=-1)  # (freq, time, 3)
        X_data_list.append(spectrograms)
        y_data_list.append(0 if condition == "healthy" else 1)

    X_data_raw = np.array(X_data_list, dtype=np.float32)  # raw spectrograms
    y_data = np.array(y_data_list, dtype=np.int32)

    print("Raw dataset shape:", X_data_raw.shape)

    # --- Extract statistical features ---
    X_features = extract_features(X_data_raw)
    print("Feature shape:", X_features.shape)

    # --- Flatten spectrograms along freq axis and combine with features ---
    n_samples, freq, time, channels = X_data_raw.shape
    X_flat = X_data_raw.reshape(n_samples, time, -1)  # shape (n_samples, time, freq*channels)

    # Expand features along time dimension
    n_feats = X_features.shape[1]
    X_features_expanded = np.repeat(X_features[:, np.newaxis, :], time, axis=1)

    # Combine spectrogram + features as last axis (channels)
    X_combined = np.concatenate([X_flat, X_features_expanded], axis=2)  # (n_samples, time, freq*channels + n_feats)
    print("Combined shape for autoencoders:", X_combined.shape)

    # --- Also prepare flat array for classical ML ---
    X_flat_ML = np.concatenate([X_data_raw.reshape(n_samples, -1), X_features], axis=1)
    print("Combined shape for classical ML:", X_flat_ML.shape)

    # Save combined dataset
    save_path = os.path.join(save_dir, "X_y_dataset.npz")
    np.savez_compressed(save_path, X_combined=X_combined, X_ML=X_flat_ML, y=y_data)
    print(f"Dataset saved: {save_path}")

    return X_combined, X_flat_ML, y_data