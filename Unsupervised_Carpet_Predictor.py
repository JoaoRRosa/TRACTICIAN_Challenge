from pydantic import BaseModel, Field,model_validator,computed_field
from typing import List
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis
from sklearn.cluster import DBSCAN 

class CarpetRegion(BaseModel): 
    start_hz: float = Field(gt = 1000, description="Start frequency in Hz") 
    end_hz: float = Field(..., description="End frequency in Hz") 

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
        return list(2.0 / self.wave_length * np.abs(fft_vals))


 
class Model: 
    def __init__(self, **params): 
        # Store hyperparameters if needed 
        self.params = params 
 
    def predict(self, wave: Wave) -> List[CarpetRegion]: 
        """ 
        Predict carpet regions from a given wave. 
        This should be implemented with actual logic. 
        """ 
        # Example placeholder 
        raise NotImplementedError("Predict method not implemented.") 