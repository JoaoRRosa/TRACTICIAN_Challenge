import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft, get_window

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
def build_dataset_from_csv(config):
    """
    Builds dataset from CSV folder using metadata from YAML config.
    Returns X_data, y_data, final_df
    """

    # -----------------------------
    # Config and paths
    # -----------------------------
    raw_folder = config["dataset"]["raw_folder"]
    output_folder = config["output"]["folder"]
    data_output = config["dataset"]["output_folder"]
    metadata_file = config["dataset"]["metadata_file"]

    os.makedirs(data_output, exist_ok=True)

    nperseg = config["dataset"]["nperseg"]
    noverlap = config["dataset"]["noverlap"]
    use_db_scale = config["dataset"]["use_db_scale"]

    # -----------------------------
    # Load metadata
    # -----------------------------
    df_meta_data = pd.read_csv(metadata_file)

    window = get_window("hann", nperseg)

    X_data = []
    y_data = []

    # -----------------------------
    # Column mapping
    # -----------------------------
    map_columns = {
        'X-Axis':'time',
        'Ch1 Y-Axis':'axisX',
        'Ch2 Y-Axis':'axisY',
        'Ch3 Y-Axis':'axisZ'
    }
    extra_cols = ['load [kw]', 'rpm', 'sensor_id', 'condition']

    # -----------------------------
    # Load CSVs
    # -----------------------------
    all_data = []
    csv_files = glob.glob(os.path.join(raw_folder, "*.csv"))

    for file_path in csv_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        df = pd.read_csv(file_path)
        df["sample_id"] = file_name

        # Step 1: rename using default mapping
        df = df.rename(columns=map_columns)

        # Step 2: rename using orientation mapping from metadata
        mask = df_meta_data["sample_id"] == file_name
        orientation_mapping = eval(df_meta_data.loc[mask, "orientation"].values[0])
        df = df.rename(columns=orientation_mapping)

        # Step 3: add extra metadata columns
        for col in extra_cols:
            df[col] = df_meta_data.loc[mask, col].values[0]

        # Step 4: determine the actual signal columns
        signal_cols = [orientation_mapping['axisX'], orientation_mapping['axisY'], orientation_mapping['axisZ']]

        df["signal_cols"] = [signal_cols]*len(df)  # optional, keep track

        all_data.append(df)

    final_df = pd.concat(all_data)
    print("Finished processing files.")

    # -----------------------------
    # STFT and feature extraction
    # -----------------------------
    for sample_id, group in final_df.groupby('sample_id'):
        group = group.sort_values('time')
        fs = estimate_sampling_frequency(group['time'].values)

        signals = [group[col].values for col in signal_cols]

        condition = group['condition'].iloc[0]
        label = 1 if condition.lower() == "healthy" else 0

        spectrograms = []
        for signal in signals:
            f, t, Zxx = stft(signal, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
            magnitude = np.abs(Zxx)
            if use_db_scale:
                magnitude = 20 * np.log10(magnitude + 1e-10)
            spectrograms.append(magnitude)

        spec_stack = np.stack(spectrograms, axis=-1)
        X_data.append(spec_stack)
        y_data.append(label)

        # Generate per-file figure
        generate_feature_figure_per_file(sample_id, signals,condition, spectrograms, f, [], output_folder)

    X_data = np.array(X_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.int32)

    # Save dataset
    np.save(os.path.join(data_output, "X_data.npy"), X_data)
    np.save(os.path.join(data_output, "y_data.npy"), y_data)

    print("Dataset saved:")
    print("X shape:", X_data.shape)
    print("y shape:", y_data.shape)

    return X_data, y_data#, final_df