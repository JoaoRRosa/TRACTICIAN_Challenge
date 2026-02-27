import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.signal import butter, filtfilt
# -------------------------
# Signal Feature Extraction
# -------------------------
def highpass_filter(signal, fs, cutoff=250, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="high")
    return filtfilt(b, a, signal)

def acceleration_to_velocity(acc, fs):
    """
    Integrate acceleration to velocity in frequency domain to avoid drift.
    """
    acc = acc - np.mean(acc)  # remove DC
    n = len(acc)
    fft_vals = np.fft.rfft(acc)
    freqs = np.fft.rfftfreq(n, 1/fs)
    vel_fft = np.zeros_like(fft_vals, dtype=np.complex64)
    for i in range(1, len(freqs)):
        vel_fft[i] = fft_vals[i] / (1j * 2 * np.pi * freqs[i])
    velocity = np.fft.irfft(vel_fft, n=n)
    return velocity

from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis, linregress
import numpy as np

# High-pass filter
def highpass_filter(signal, fs, cutoff=250, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="high")
    return filtfilt(b, a, signal)

def extract_features_from_signals(signals, fs, rpm, time=None):
    """
    Extract features from a list of 1D signals [horizontal, axial, vertical].
    Adds velocity-based features: RMS, slope (trend), R^2, peak-to-peak.
    
    Args:
        signals: list of 1D numpy arrays (acceleration signals)
        fs: sampling frequency
        rpm: operational RPM
        time: optional time vector for slope calculation (default linear)
    Returns:
        feats: 1D numpy array of features
    """
    feats = []
    
    if time is None:
        n = len(signals[0])
        time = np.arange(n) / fs

    for sig in signals:
        # Acceleration features
        rms = np.sqrt(np.mean(sig**2))
        rms_hp = np.sqrt(np.mean(highpass_filter(sig, fs)**2))
        peak = np.max(np.abs(sig))
        crest = peak / rms if rms != 0 else 0
        zc = np.sum(np.diff(np.sign(sig)) != 0) / len(sig)
        k = kurtosis(sig)
        f_op = rpm / 60.0  # Hz
        
        # Velocity signal
        vel = np.gradient(sig, time)  # numerical derivative
        vel_rms = np.sqrt(np.mean(vel**2))
        vel_ptp = np.ptp(vel)  # peak-to-peak
        # Velocity trend
        slope, intercept, r_value, p_value, std_err = linregress(time, vel)
        vel_slope = slope
        vel_r2 = r_value**2  # confidence in linear trend

        feats.extend([
            rms, rms_hp, crest, zc, k, f_op,
            vel_rms, vel_ptp, vel_slope, vel_r2
        ])
    
    return np.array(feats, dtype=np.float32)

# -------------------------
# Main Dataset Builder
# -------------------------
def build_feature_dataset_from_csv(folder_path, metadata_file, save_dir="outputs/feature_dataset"):
    os.makedirs(save_dir, exist_ok=True)

    df_meta = pd.read_csv(metadata_file)
    map_columns = {
        'X-Axis': 'time',
        'Ch1 Y-Axis': 'axisX',
        'Ch2 Y-Axis': 'axisY',
        'Ch3 Y-Axis': 'axisZ'
    }

    all_rows = []
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    for file_path in csv_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        df = pd.read_csv(file_path)
        df["sample_id"] = file_name
        df = df.rename(columns=map_columns)

        mask = df_meta["sample_id"] == file_name
        orientation_mapping = eval(df_meta.loc[mask, "orientation"].values[0])
        df = df.rename(columns=orientation_mapping)

        # Fill metadata
        for col in ["load [kw]", "rpm", "sensor_id", "condition"]:
            df[col] = df_meta.loc[mask, col].values[0]

        # Compute sampling freq
        df = df.sort_values("time")
        dt = df["time"].diff().dropna()
        fs = 1.0 / np.median(dt)
        rpm = df["rpm"].iloc[0]
        condition = df["condition"].iloc[0]

        signals = [
            df["horizontal"].values,
            df["axial"].values,
            df["vertical"].values
        ]

        features = extract_features_from_signals(signals, fs, rpm)
        all_rows.append(np.concatenate([[0 if condition=="healthy" else 1], features]))  # label first

        # -------------------------
        # Plot signals + velocity + FFT for this sample
        # -------------------------
        fig, axs = plt.subplots(3, 3, figsize=(18, 12))  # 3 signals x 3 columns (Velocity, Acc Time, FFT)
        axes_names = ["Horizontal", "Axial", "Vertical"]

        for i, sig in enumerate(signals):
            # Compute velocity
            vel = acceleration_to_velocity(sig, fs)

            # Velocity time-domain
            axs[i, 0].plot(df["time"], vel, color='green')
            axs[i, 0].set_title(f" {axes_names[i]} (Velocity)")
            axs[i, 0].set_xlabel("Time [s]")
            axs[i, 0].set_ylabel("Velocity [unit/s]")

            # Acceleration time-domain
            axs[i, 1].plot(df["time"], sig, color='blue')
            axs[i, 1].set_title(f"{file_name} - {condition} - {axes_names[i]} (Acceleration)")
            axs[i, 1].set_xlabel("Time [s]")
            axs[i, 1].set_ylabel("Amplitude")

            # FFT of acceleration
            n = len(sig)
            freqs = np.fft.rfftfreq(n, d=dt.median())
            fft_vals = np.fft.rfft(sig)
            amp = np.abs(fft_vals) / n
            axs[i, 2].plot(freqs, amp)
            axs[i, 2].set_title(f"{file_name} - {condition} - {axes_names[i]} (FFT)")
            axs[i, 2].set_xlabel("Frequency [Hz]")
            axs[i, 2].set_ylabel("Amplitude")
            # Highlight operational frequency
            f_op = rpm / 60.0
            axs[i, 2].axvline(f_op, color="red", linestyle="--", label="Operating freq")
            axs[i, 2].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{file_name}_signals_velocity_fft.png"), dpi=300)
        plt.close(fig)

    # Final dataset
    all_rows = np.array(all_rows, dtype=np.float32)
    X = all_rows[:, 1:]  # features
    y = all_rows[:, 0].astype(int)

    # Save dataset
    save_path = os.path.join(save_dir, "X_y_features.npz")
    np.savez_compressed(save_path, X=X, y=y)
    print(f"Feature dataset saved: {save_path}, X shape: {X.shape}, y shape: {y.shape}")

    return X, y