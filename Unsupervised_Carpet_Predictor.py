from pydantic import BaseModel, Field,model_validator,computed_field
from typing import List,Dict,Tuple
import os
import glob
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis
from sklearn.cluster import DBSCAN
from Wave_utils import Wave,Wave_filter,Region,CarpetRegion 


    
 
class Model: 
    def __init__(self, **params): 
        # Store hyperparameters if needed 
        self.params = params
 
    def predict(self, wave: Wave, output_file_fig : str = '', output_file_features = '') -> List[CarpetRegion]: 
        """ 
        Predict carpet regions from a given wave. 
        This should be implemented with actual logic. 
        """ 
        wave_filtered = self.apply_highpass_filter(wave, cutoff=1000)
        carpet_regions = self.detect_DBSCAN_clusters(wave_filtered)
        final_carpet_regions,rms_values_per_region = self.apply_RMS_filter(wave,carpet_regions)

        if output_file_fig != '':
            self.plot_unsupervised_model_steps(output_file_fig,wave,carpet_regions,rms_values_per_region,final_carpet_regions)
        
        if output_file_fig != '':
            complementary_regions = self.get_complementary_regions(wave,carpet_regions)
            regions = carpet_regions + complementary_regions
            self.extract_features(output_file_features,wave,regions,final_carpet_regions)
         
        return final_carpet_regions
    
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
    
    def detect_DBSCAN_clusters(self, wave: Wave) -> List[CarpetRegion]:
        """Applies a Density Based Spatial Clustering model to obtain clusters of points higly concentrated"""

        # Highperparameters of the DBSCAN
        percentile_amplitudes_threshold = self.params['percentile_amplitudes_threshold']
        dbscan_eps = self.params['dbscan_eps']
        dbscan_min_samples = self.params['dbscan_min_samples']

        amplitudes = np.array(wave.amplitudes)
        freqs = np.array(wave.frequencies)

        threshold = np.percentile(amplitudes, percentile_amplitudes_threshold)
        mask = amplitudes >= threshold
        X = np.column_stack([freqs[mask], amplitudes[mask]])
        if len(X) == 0:
            print(f'Error the the amlpitudes above percentile {percentile_amplitudes_threshold} dont exist')
            return []
        clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(X)
        labels = clustering.labels_
        regions = []
        for label in set(labels):
            if label == -1:
                continue
            cluster_freqs = X[labels == label, 0]
            #regions.append((cluster_freqs.min(), cluster_freqs.max()))
            carpet_start = cluster_freqs.min() if cluster_freqs.min()>1000 else 1001
            carpet_end = cluster_freqs.max()
            if carpet_end<1000:
                continue
            regions.append(CarpetRegion(start_hz = carpet_start,end_hz = carpet_end))
        return sorted(regions, key=lambda region: region.start_hz)
    
    def apply_RMS_filter(self, wave: Wave, regions : List[CarpetRegion]) -> Tuple[List[CarpetRegion], List[float]]:
        """Filters the CarpetRegions selected based on the ammount of energy it posess compared to the overall energy from the signal"""

        #Hiperparameter of the RMS filter
        percentage_overall_RMS = self.params['percentage_overall_RMS']/(100 * len(regions))

        overall_rms = self.root_mean_squared(wave)
        rms_values_per_region = []
        carpet_regions_list = []
        for region in regions:
            f_low = region.start_hz
            f_high = region.end_hz
            filtered_wave = self.apply_bandpass_filter(wave, f_low, f_high)
            region_rms = self.root_mean_squared(filtered_wave)
            if region_rms>=(overall_rms*percentage_overall_RMS):
                carpet_regions_list.append(region)

            rms_values_per_region.append(region_rms)

        return carpet_regions_list,rms_values_per_region
    
    def root_mean_squared(self,wave:Wave) -> float:
        """Calculates the root mean square of the wave signal, this value simbolyzes the energy of the signal"""
        return np.sqrt(np.mean(np.array(wave.signal)**2))
    
    def extract_features(self,save_path : str, wave: Wave, regions : List[Region],carpet_regions: List[CarpetRegion]) -> List[Dict]:
        """Exctracts features from the carpet regions to train a supervised model"""
        amplitudes = wave.amplitudes
        freqs = wave.frequencies
        # Sort subregions by start value
        regions = sorted(regions, key=lambda region: region.start_hz)

        feature_rows = []
        for region in regions:
            f_low = region.start_hz
            f_high = region.end_hz
            #mask = (freqs >= f_low) & (freqs <= f_high)
            #band_amp = amplitudes[mask]
            wave_filtered = self.apply_bandpass_filter(wave,f_low,f_high)
            band_amp = np.array(wave.amplitudes)
            if len(band_amp) == 0:
                continue
            bandwidth = f_high - f_low
            spectral_energy = np.sum(band_amp**2)
            mean_amp = np.mean(band_amp)
            max_amp = np.max(band_amp)
            signal = np.array(wave_filtered.signal)
            rms = np.sqrt(np.mean(signal**2))
            std_val = np.std(signal)
            kurt_val = kurtosis(signal, fisher=False)
            label = 1 if region in carpet_regions else 0
            
            feature_rows.append({
                "region":f"{round(f_low,1)}-{round(f_high,1)}",
                "f_low": f_low,
                "f_high": f_high,
                "bandwidth": bandwidth,
                "spectral_energy": spectral_energy,
                "mean_amplitude": mean_amp,
                "max_amplitude": max_amp,
                "rms": rms,
                "std": std_val,
                "kurtosis": kurt_val,
                "label": label})

        final_df = pd.DataFrame(feature_rows)
        file_name = os.path.splitext(save_path)[0]
        figure_name = os.path.join(f'{file_name}.png')
        final_df.to_csv(save_path, index=False)
        print("\nDataset saved:", save_path)

        self.plot_features_per_region(figure_name,final_df,"region","label")
        return feature_rows
    
    def plot_features_per_region(self,save_path,df,region_col,label_col,positive_color="red",
                               negative_color="blue",figsize_per_plot=(10, 4)):
        """
        Creates one subplot per feature.
        Bars are colored based on label (0/1).
        Saves the resulting figure.
        """

        # Identify feature columns
        feature_cols = [col for col in df.columns 
                        if col not in [region_col, label_col]]

        n_features = len(feature_cols)

        fig, axes = plt.subplots(
            nrows=n_features,
            ncols=1,
            figsize=(figsize_per_plot[0], figsize_per_plot[1] * n_features),
            sharex=True
        )

        # If only one feature, axes is not iterable
        if n_features == 1:
            axes = [axes]

        colors = df[label_col].map({
            1: positive_color,
            0: negative_color
        })

        for ax, feature in zip(axes, feature_cols):
            ax.bar(
                df[region_col],
                df[feature],
                color=colors
            )
            ax.set_title(feature)
            ax.set_ylabel("Value")
            ax.grid(axis="y", linestyle="--", alpha=0.4)

        axes[-1].set_xlabel("Region")
        plt.xticks(rotation=45)

        plt.tight_layout()

        # Create directory if needed
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None

        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Figure saved to: {save_path}")
        pass
   
    def get_complementary_regions(self,wave : Wave ,regions : List[CarpetRegion]) -> List[Region]:
        big_region = Region(start_hz=wave.min_frequency,end_hz=wave.max_frequency)
        #start, end = wave.min_frequency,wave.max_frequency
        
        if big_region in regions:
            return [big_region]
        
        # Sort subregions by start value
        regions = sorted(regions, key=lambda region: region.start_hz)
        
        complementary_regions = []
        current = big_region.start_hz
        
        for region in regions:
            s, e = region.start_hz,region.end_hz
            # Ignore subregions outside big_region
            if e <= big_region.start_hz or s >= big_region.end_hz:
                continue
            
            # Clip to big_region bounds
            s = max(s, big_region.start_hz)
            e = min(e, big_region.end_hz)
            
            if s > current:
                complementary_regions.append(Region(start_hz = current, end_hz = s))
            
            current = max(current, e)
        
        if current < big_region.end_hz:
            complementary_regions.append(Region(start_hz = current, end_hz = big_region.end_hz))
        
        return complementary_regions

    def plot_unsupervised_model_steps(self,save_path : str,wave: Wave, regions: List[CarpetRegion],
                                      rms_values_per_region: List[float], final_regions : List[CarpetRegion]):
        
        fig, axs = plt.subplots(4, 1, figsize=(14, 16))
        fig.suptitle(os.path.basename(save_path), fontsize=16)

        freqs = wave.frequencies
        amplitudes = wave.amplitudes

        # 1 Raw FFT
        axs[0].plot(freqs, amplitudes)
        axs[0].set_title("1) Raw FFT")

        # 2 DBSCAN clusters
        axs[1].plot(freqs, amplitudes)
        for region in regions:
            axs[1].axvspan(region.start_hz, region.end_hz, color='orange', alpha=0.3)
        axs[1].set_title("2) DBSCAN Clusters")
        axs[1].legend()

        # 3 RMS filter plot
        labels_plot = [f"{round(region.start_hz,0)}-{round(region.end_hz,0)}" for region in regions]
        full_signal_rms = self.root_mean_squared(wave)
        threshold = self.params['percentage_overall_RMS'] * full_signal_rms/(100 * len(regions))
        axs[2].bar(labels_plot + ["Full signal"],rms_values_per_region + [full_signal_rms])
        axs[2].axhline(y=threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
        axs[2].set_title("3) RMS filter selection")
        axs[2].legend()
        # 3 Global RMS histogram
        #axs[2].hist(all_rms, bins=40, color='gray')
        #axs[2].axvline(global_threshold, color='red', linewidth=2)
        #axs[2].set_title("3) Global RMS Distribution")

        # 4 Final selected regions
        axs[3].plot(freqs, amplitudes)
        for region in final_regions:
            axs[3].axvspan(region.start_hz, region.end_hz, color='red', alpha=0.4)
        axs[3].set_title("4) Final Selected Regions")
        axs[3].legend()

        plt.tight_layout(rect=[0,0,1,0.97])
        #save_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(file))[0]}_pipeline.png")
        #save_path = f"{os.path.splitext(os.path.basename(file))[0]}_pipeline.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved pipeline figure: {save_path}")

if __name__ == "__main__":

    INPUT_FOLDER = 'part_1'
    OUTPUT_FOLDER = 'outputs/pydantic_model'
    FEATURES_FOLDER = 'outputs/pydantic_model/features'

    os.makedirs(OUTPUT_FOLDER,exist_ok=True)
    os.makedirs(FEATURES_FOLDER,exist_ok=True)
    csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))

    # Load YAML from file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    params = config['model']['params']    
    model = Model(**params)

    for file in csv_files:

        print(f"Processing: {file}")

        df = pd.read_csv(file)
        time = df.iloc[:, 0].to_numpy().copy()
        signal = df.iloc[:, 1].to_numpy().copy()
        signal = (signal - np.mean(signal))#*9.81
        
        wave = Wave(time = time,signal = signal)
        save_path_pipeline = os.path.join(OUTPUT_FOLDER,f"{os.path.splitext(os.path.basename(file))[0]}_pipeline.png")
        save_path_features = os.path.join(FEATURES_FOLDER,f"{os.path.splitext(os.path.basename(file))[0]}_features.csv")

        carpet_region_list = model.predict(wave,output_file_fig=save_path_pipeline,output_file_features=save_path_features)