import os
import yaml
import numpy as np
from joblib import load as joblib_load
import pandas as pd
from sklearn.metrics import accuracy_score
import glob

from Wave_utils import Wave, extract_features_from_signals,plot_waves
import matplotlib.pyplot as plt

def plot_features_with_test_predictions(
    X_train_features, y_train,
    X_test_features, y_test_pred,
    feature_names=None,
    save_dir="outputs/feature_analysis",
    title="Feature comparison train vs predicted test"
):
    """
    Plots aggregated features by class (train) and predicted labels (test).
    
    Parameters
    ----------
    X_train_features : np.ndarray
        Shape (n_train_samples, n_features)
    y_train : np.ndarray
        Labels for training data (0=healthy, 1=faulty)
    X_test_features : np.ndarray
        Shape (n_test_samples, n_features)
    y_test_pred : np.ndarray
        Predicted labels for test data (0=healthy, 1=faulty)
    feature_names : list of str
        Names of features (length must match n_features)
    save_dir : str
        Folder to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    n_features = X_train_features.shape[1]
    
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(n_features)]*3
    
    # Convert to arrays if needed
    X_train_features = np.array(X_train_features)
    y_train = np.array(y_train)
    X_test_features = np.array(X_test_features)
    y_test_pred = np.array(y_test_pred)

    # Aggregate features: mean per class
    train_mean = []
    test_mean = []
    classes = [0, 1]
    for cls in classes:
        train_mean.append(X_train_features[y_train==cls].mean(axis=0))
        test_mean.append(X_test_features[y_test_pred==cls].mean(axis=0))

    train_mean = np.array(train_mean)
    test_mean = np.array(test_mean)

    # Plot all features in a single figure with subplots
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axs = axs.flatten()

    for i in range(n_features):
        ax = axs[i]
        # Training data
        ax.bar(np.arange(len(classes))-0.2, train_mean[:, i], width=0.4, label="Train")
        # Test predictions
        ax.bar(np.arange(len(classes))+0.2, test_mean[:, i], width=0.4, label="Test Pred")
        ax.set_xticks([0,1])
        ax.set_xticklabels(["Healthy","Faulty"])
        ax.set_title(feature_names[i])
        ax.grid(True)
        if i==0:
            ax.legend()

    plt.suptitle(title)
    plt.tight_layout(rect=[0,0,1,0.95])
    save_path = os.path.join(save_dir, "features_train_vs_test_pred.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Feature comparison plot saved: {save_path}")

class LoosenessModel:

    def __init__(self, **params):
        self.params = params
        self.load_model()

    def load_model(self):
        model_path = self.params['predictor']['model_path']
        self.model = joblib_load(model_path)
        scaler_path = self.params['predictor']['scaler_path']
        self.scaler = joblib_load(scaler_path)
            

    def score(self, wave_hor: Wave, wave_ver: Wave, wave_axi: Wave) -> float:

        X = extract_features_from_signals([wave_hor, wave_axi, wave_ver])
        # Ensure X is 2D: (1, n_features)
        X = X.reshape(1, -1)
        #print("X_test min/max:", X.min(), X.max())
        X_scaled = self.scaler.transform(X)
        #print("X_test min/max:", X_scaled.min(), X_scaled.max())
        score = self.model.predict_proba(X_scaled)[0][1]  # probability of class 1
        return float(score)

    def predict(self, wave_hor: Wave, wave_ver: Wave, wave_axi: Wave) -> bool:

        score = self.score(wave_hor, wave_ver, wave_axi)
        threshold = self.params['predictor']['threshold']

        return bool(score > threshold)


def load_config(config_path="Part2_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":

    # Load configuration
    config = load_config("part_2_code/Part2_config.yaml")

    # Initialize model
    model = LoosenessModel(**config)
    folder_path = config['test']['data']
    metadata_file = config['test']['metadata_file']


    # Load metadata
    df_meta_data = pd.read_csv(metadata_file)

    # Column mapping
    map_columns = {
        't': 'time',
        'x': 'axisX',
        'y': 'axisY',
        'z': 'axisZ'
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
        df["rpm"] = df_meta_data.loc[mask, "rpm"].values[0]

        all_data.append(df)


    # Combine all files
    final_df = pd.concat(all_data)
    print("Finished processing files.")

    # --- Compute STFT spectrograms ---
    X_test = []
    y_pred_test = []

    for sample_id, group in final_df.groupby("sample_id"):
        group = group.sort_values("time")

        signals = [
            group["horizontal"].values,
            group["axial"].values,
            group["vertical"].values
        ]
        #condition = 0 if group["condition"].iloc[0]=='healthy' else 1


        # Load waves from paths in config
        wave_hor = Wave(time = group['time'].values.tolist(),signal = group['horizontal'].values.tolist())
        wave_ver = Wave(time = group['time'].values.tolist(),signal = group['vertical'].values.tolist())
        wave_axi = Wave(time = group['time'].values.tolist(),signal = group['axial'].values.tolist())

        # Predict looseness
        prediction = model.predict(wave_hor, wave_ver, wave_axi)
        score = model.score(wave_hor, wave_ver, wave_axi)

        print("Looseness score:", score)
        print("Looseness detected:", prediction)
        #y.append(condition)
        X_test.append(extract_features_from_signals([wave_hor, wave_ver, wave_axi]))
        y_pred_test.append(prediction)

        fo = group["rpm"].iloc[0]
        condition = 'Loseness' if prediction else 'Healthy'
        plot_waves([wave_hor,wave_axi,wave_ver],fo,condition,sample_id,'test/waves')

    data = np.load(config['predictor']['Xy_folder'],allow_pickle=True)

    # Access arrays
    X = data['X']
    y = data['y']

    plot_features_with_test_predictions(X, y,X_test, y_pred_test)#
                                        #feature_names =["RMS"]*3,"RMS_HP","CrestFactor","ZeroCrossings","Kurtosis",'frequency'])



        
