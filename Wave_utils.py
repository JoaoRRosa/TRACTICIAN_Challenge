from pydantic import BaseModel, Field,model_validator,computed_field
from typing import List,Dict,Tuple
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.signal import stft, get_window
from scipy.stats import skew, kurtosis, linregress
import matplotlib.pyplot as plt
import os

class Region(BaseModel): 
    start_hz: float = Field(ge = 0, description="Start frequency in Hz") 
    end_hz: float = Field(..., description="End frequency in Hz") 

class CarpetRegion(Region): 
    start_hz: float = Field(gt = 1000, description="Start frequency in Hz") 
     
class Wave(BaseModel): 
    time: List[float] = Field(min_length=1, description="Time points of the wave") 
    signal: List[float] = Field(min_length=1, description="Signal values")

    @model_validator(mode="after")
    def check_time_and_signal(self):
        """Checks if time and signal have the same length, if there are NANs or Infs, and if time data is growing"""

        # Checks if time and signal have the same length
        if len(self.time) != len(self.signal):
            raise ValueError("Time and signal data do not have the same lenghts!")
        
        time_arr = np.array(self.time, dtype=float)
        signal_arr = np.array(self.signal, dtype=float)

        # Checks if there is NAN or Inf
        if not np.all(np.isfinite(time_arr)):
            raise ValueError("Time array contains NaN or infinite values")
        
        if not np.all(np.isfinite(signal_arr)):
            raise ValueError("Signal array contains NaN or infinite values")

        # Checks if time is strictly growing
        #if np.any(np.diff(time_arr) <= 0):
            #raise ValueError("Time values must be strictly increasing")

        return self

    @computed_field(description = "Length of the signal (should be equal to length of time)")
    @property
    def wave_length(self) -> int:
        """Length of the signal (should be equal to length of time)"""
        return len(self.signal)

    @computed_field(description ="Data collection frequency calculated from the average time delta between points")
    @property
    def wave_frequency(self) -> float:
        """Data collection frequency calculated from the average time delta between points"""
        dt = np.mean(np.diff(self.time))
        return 1.0 / dt
    
    @computed_field(description ="Nyquist frequency of the wave")
    @property
    def nyquist_frequency(self) -> float:
        """Nyquist frequency of the wave calculated from the average time delta between points"""
        return self.wave_frequency / 2.0

    @computed_field(description="Frequencies calculated from the fast fourier transform")
    @property
    def frequencies(self) -> List[float]:
        """Frequencies calculated from the fast fourier transform"""
        dt = np.mean(np.diff(self.time))
        return list(np.fft.rfftfreq(self.wave_length, dt))

    @computed_field(description="Amplitudes calculated from the fast fourier transform")
    @property
    def amplitudes(self) -> List[float]:
        """Amplitudes calculated from the fast fourier transform"""
        dt = np.mean(np.diff(self.time))
        fft_vals = np.fft.rfft(self.signal)
        return list((2.0 / self.wave_length) * np.abs(fft_vals))
    
    @computed_field(description="Maximum amplitude from the fast fourier transform")
    @property
    def max_amplitude(self) -> float:
        """Maximum amplitude from the fast fourier transform"""
        return max(self.amplitudes)
    
    @computed_field(description="Minimum amplitude from the fast fourier transform")
    @property
    def min_amplitude(self) -> float:
        """Minimum amplitude from the fast fourier transform"""
        return min(self.amplitudes)
    
    @computed_field(description="Maximum amplitude from the fast fourier transform")
    @property
    def max_frequency(self) -> float:
        """Maximum frequency from the fast fourier transform"""
        return max(self.frequencies)
    
    @computed_field(description="Minimum amplitude from the fast fourier transform")
    @property
    def min_frequency(self) -> float:
        """Minimum frequency from the fast fourier transform"""
        return min(self.frequencies)
    
class Wave_filter:

    def apply_highpass_filter(self, wave: Wave, cutoff : float, order: int = 4) -> Wave:
        """Applies a High-Pass filter to the wave (Carpet noises only happens above 1kHz)"""
        nyq = wave.nyquist_frequency
        sos = butter(order, cutoff / nyq, btype='highpass', output='sos')
        #return Wave(time = wave.time ,signal = sosfiltfilt(sos, wave.signal))
        return wave.model_copy(update={"signal": sosfiltfilt(sos, wave.signal)})

    def apply_bandpass_filter(self, wave: Wave, f_low : float, f_high : float, order=4) -> Wave:
        """Applies a Band-Pass filter to the wave, is used here to filter each region that the DBSCAN model selected"""
        nyq = wave.nyquist_frequency
        f_low = max(f_low, 1)
        f_high = min(f_high, nyq - 1)
        if f_low >= f_high:
            return np.zeros_like(wave.signal)
        sos = butter(order, [f_low/nyq, f_high/nyq], btype='bandpass', output='sos')
        filtered = sosfiltfilt(sos, wave.signal)
        trim = int(0.05 * len(filtered))
        if trim > 0:
            filtered = filtered[trim:-trim]

        return wave.model_copy(update={"signal": filtered})

def extract_features_from_signals(waves:List[Wave], cutoff=250):
    """
    Extract features from a list of 1D signals [horizontal, axial, vertical]
    """
    feats = []
    filter = Wave_filter()

    for wave in waves:
        signal = np.array(wave.signal)
        time = np.array(wave.time)

        rms = np.sqrt(np.mean(signal**2))
        filtered_wave = filter.apply_highpass_filter(wave,cutoff=cutoff)
        rms_hp = np.sqrt(np.mean(np.array(filtered_wave.signal)**2))
        peak = np.max(np.abs(signal))
        crest = peak / rms if rms != 0 else 0
        zc = np.sum(np.diff(np.sign(signal)) != 0) / len(signal)
        k = kurtosis(signal)
        # Add operational frequency
        #f_op = rpm / 60.0
        f_op = estimate_operational_frequency_from_signals(waves)

        # Velocity features
        # Velocity signal
        vel = np.gradient(signal, time)  # numerical derivative
        vel_rms = np.sqrt(np.mean(vel**2))
        vel_ptp = np.ptp(vel)  # peak-to-peak
        # Velocity trend
        slope, intercept, r_value, p_value, std_err = linregress(time, vel)
        vel_slope = slope
        vel_r2 = r_value**2  # confidence in linear trend

        feats.extend([
            rms, rms_hp, crest, zc, k, #f_op,
            vel_rms, vel_ptp, vel_slope, vel_r2
        ])

    return np.array(feats, dtype=np.float32)

def compute_spectrograms(waves: list[Wave], fs=1.0, window='hann', nperseg=256, noverlap=None, use_db_scale=True):
    """
    Compute STFT spectrograms for a batch of signals.
    
    Parameters
    ----------
    signals : list or np.ndarray
        List or array of 1D signals (each signal = 1D array).
    fs : float
        Sampling frequency of the signals.
    window : str or tuple or array_like
        Desired window to use for STFT.
    nperseg : int
        Length of each segment for STFT.
    noverlap : int or None
        Number of points to overlap between segments. If None, defaults to nperseg//2.
    use_db_scale : bool
        If True, convert magnitude to decibel (dB) scale.
    
    Returns
    -------
    spectrograms : list of np.ndarray
        List of spectrogram magnitude arrays (frequency x time) for each signal.
    freqs : np.ndarray
        Array of sample frequencies.
    times : np.ndarray
        Array of segment times.
    """
    if noverlap is None:
        noverlap = nperseg // 2

    spectrograms = []
    freqs, times = None, None

    for wave in waves:
        fs = estimate_sampling_frequency(wave)
        f, t, Zxx = stft(wave.signal, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(Zxx)
        if use_db_scale:
            mag = 20 * np.log10(mag + 1e-10)  # add small epsilon to avoid log(0)
        spectrograms.append(mag)

        # Store frequency and time arrays from first signal
        if freqs is None:
            freqs, times = f, t

    return spectrograms, freqs, times

def estimate_sampling_frequency(wave: Wave):
    dt = np.diff(wave.time)
    median_dt = np.median(dt)
    return 1.0 / median_dt

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

def estimate_operational_frequency_from_signals(waves : List[Wave], min_freq=1.0, max_freq=None):


    """
    Estimate operational frequency from multiple signals (e.g., horizontal, axial, vertical)
    by finding the dominant frequency in each direction and averaging.

    Parameters
    ----------
    signals : list of 1D numpy arrays
        Signals in different directions.
    fs : float
        Sampling frequency (Hz).
    min_freq : float
        Minimum frequency to consider (Hz) to avoid low-frequency noise.
    max_freq : float or None
        Maximum frequency to consider (Hz). Defaults to Nyquist.

    Returns
    -------
    f_op : float
        Estimated operational frequency (Hz)
    """
    if max_freq is None:
        max_freq = waves[0].wave_frequency / 2

    dominant_freqs = []

    for wave in waves:
        n = wave.wave_length
        signal = np.array(wave.signal)
        n = len(signal)
        fft_vals = np.fft.rfft(signal)
        fft_freqs = np.fft.rfftfreq(n, 1/wave.wave_frequency)
        magnitude = np.abs(fft_vals)

        # Only consider frequencies within the specified range
        valid_idx = (fft_freqs >= min_freq) & (fft_freqs <= max_freq)
        fft_freqs = fft_freqs[valid_idx]
        magnitude = magnitude[valid_idx]

        # Find frequency with maximum amplitude
        dom_freq = fft_freqs[np.argmax(magnitude)]
        dominant_freqs.append(dom_freq)

    # Average dominant frequency across all signals
    f_op = np.mean(dominant_freqs)
    return f_op

def plot_waves(waves: List[Wave], fo, condition, file_name, save_dir):
    """
    Plots velocity, acceleration, and FFT for each wave axis.

    Parameters
    ----------
    waves : List[Wave]
        List containing Wave objects (one per axis).
    fo : float
        Operational frequency (Hz).
    condition : str
        Condition label.
    file_name : str
        Base file name.
    save_dir : str
        Directory to save the figure.
    fs : float
        Sampling frequency.
    """

    fig, axs = plt.subplots(3, 3, figsize=(18, 14))  # 3 axes × 3 columns
    axes_names = ["Horizontal", "Axial", "Vertical"]

    for i, wave in enumerate(waves):

        # -----------------------
        # Compute velocity
        # -----------------------
        fs = wave.wave_frequency
        vel = acceleration_to_velocity(wave.signal, fs)

        # -----------------------
        # Velocity (time-domain)
        # -----------------------
        axs[i, 0].plot(wave.time, vel, color="green")
        axs[i, 0].set_title(f"{axes_names[i]} (Velocity)")
        axs[i, 0].set_xlabel("Time [s]")
        axs[i, 0].set_ylabel("Velocity")

        # -----------------------
        # Acceleration (time-domain)
        # -----------------------
        axs[i, 1].plot(wave.time, wave.signal, color="blue")
        axs[i, 1].set_title(f"{axes_names[i]} (Acceleration)")
        axs[i, 1].set_xlabel("Time [s]")
        axs[i, 1].set_ylabel("Amplitude")

        # -----------------------
        # FFT (from acceleration)
        # -----------------------
        n = len(wave.signal)
        freqs = np.fft.rfftfreq(n, d=1/fs)
        fft_vals = np.fft.rfft(wave.signal)
        amp = np.abs(fft_vals) / n

        axs[i, 2].plot(freqs, amp)
        axs[i, 2].set_title(f"{axes_names[i]} (FFT)")
        axs[i, 2].set_xlabel("Frequency [Hz]")
        axs[i, 2].set_ylabel("Amplitude")

        # Highlight operational frequency
        axs[i, 2].axvline(fo, color="red", linestyle="--", label="Operating freq")
        axs[i, 2].legend()

    fig.suptitle(f"{file_name} - {condition} - Operational Frequency: {fo:.2f} Hz", fontsize=16)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{file_name}_signals_velocity_acc_fft.png"), dpi=300)
    plt.close(fig)