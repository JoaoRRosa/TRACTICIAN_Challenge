from pydantic import BaseModel, Field,model_validator,computed_field
from typing import List,Dict,Tuple
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.signal import stft, get_window
from scipy.stats import skew, kurtosis


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

def extract_features_from_signals(waves:List[Wave], rpm, cutoff=250):
    """
    Extract features from a list of 1D signals [horizontal, axial, vertical]
    """
    feats = []
    filter = Wave_filter

    for wave in waves:
        signal = np.array(wave.signal)
        rms = np.sqrt(np.mean(signal**2))
        rms_hp = np.sqrt(np.mean(filter.apply_highpass_filter(signal, wave.wave_frequency,cutoff=cutoff)**2))
        peak = np.max(np.abs(signal))
        crest = peak / rms if rms != 0 else 0
        zc = np.sum(np.diff(np.sign(signal)) != 0) / len(signal)
        k = kurtosis(signal)
        # Add operational frequency
        f_op = rpm / 60.0
        feats.extend([rms, rms_hp, crest, zc, k, f_op])
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