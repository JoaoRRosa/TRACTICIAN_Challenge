from pydantic import BaseModel, Field,model_validator,computed_field
from typing import List,Dict,Tuple
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis

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
        if np.any(np.diff(time_arr) <= 0):
            raise ValueError("Time values must be strictly increasing")

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


