import numpy as np
# models.py
# models.py
import math
import tensorflow as tf
from tensorflow.keras import layers, models

# -------------------------------
# 1D CNN Autoencoder
# -------------------------------
def build_1d_cnn_autoencoder(input_shape, filters=[64,32], optimizer='adam'):
    """
    1D CNN Autoencoder with automatic handling of sequence length
    input_shape: (timesteps, features)
    filters: list of Conv1D filters
    optimizer: optimizer name or object
    """
    inp = layers.Input(shape=input_shape)
    x = inp

    pool_count = len(filters)
    original_len = input_shape[0]

    # Encoder
    for f in filters:
        x = layers.Conv1D(f, 3, padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(2, padding='same')(x)

    # Flatten
    encoded = layers.Flatten()(x)

    # Compute final time dimension after pooling
    down_len = math.ceil(original_len / (2 ** pool_count))
    x = layers.Dense(down_len * filters[-1], activation='relu')(encoded)
    x = layers.Reshape((down_len, filters[-1]))(x)

    # Decoder
    for f in reversed(filters):
        x = layers.UpSampling1D(2)(x)
        x = layers.Conv1D(f, 3, padding='same', activation='relu')(x)

    # Final layer to match original feature dimension
    decoded = layers.Conv1D(input_shape[1], 3, padding='same', activation='linear')(x)

    # Crop if upsampled output is longer than original
    def crop_to_original(x_dec):
        current_len = tf.shape(x_dec)[1]
        return x_dec[:, :original_len, :]

    decoded = layers.Lambda(crop_to_original)(decoded)

    model = models.Model(inp, decoded)
    model.compile(optimizer=optimizer, loss='mse')
    return model


# -------------------------------
# LSTM Autoencoder
# -------------------------------
def build_lstm_autoencoder(input_shape, latent_dim=64, optimizer='adam'):
    """
    LSTM Autoencoder
    input_shape: (timesteps, features)
    latent_dim: dimension of encoded representation
    optimizer: optimizer name or object
    """
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(latent_dim, return_sequences=False)(inp)
    x = layers.RepeatVector(input_shape[0])(x)
    decoded = layers.LSTM(input_shape[1], return_sequences=True)(x)

    model = models.Model(inp, decoded)
    model.compile(optimizer=optimizer, loss='mse')
    return model


# -------------------------------
# 2D CNN Autoencoder
# -------------------------------
def build_2d_cnn_autoencoder(input_shape, filters=[32,16], latent_dim=128, optimizer='adam'):
    """
    2D CNN Autoencoder
    input_shape: (freq_bins, time_steps, channels)
    filters: list of Conv2D filters
    latent_dim: dimension of encoded representation
    optimizer: optimizer name or object
    """
    inp = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(filters[0], 3, activation='relu', padding='same')(inp)
    x = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Conv2D(filters[1], 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2, padding='same')(x)

    x = layers.Flatten()(x)
    encoded = layers.Dense(latent_dim, activation='relu')(x)

    # Decoder
    freq_dec = input_shape[0] // 4
    time_dec = input_shape[1] // 4
    channels = input_shape[2]

    x = layers.Dense(freq_dec * time_dec * filters[1], activation='relu')(encoded)
    x = layers.Reshape((freq_dec, time_dec, filters[1]))(x)

    x = layers.Conv2DTranspose(filters[1], 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(filters[0], 3, strides=2, activation='relu', padding='same')(x)
    decoded = layers.Conv2D(channels, 3, activation='linear', padding='same')(x)

    model = models.Model(inp, decoded)
    model.compile(optimizer=optimizer, loss='mse')
    return model

import numpy as np

def determine_ae_threshold_max_division(errors_train, y_train, num_thresholds=1000):
    """
    Determine the best threshold for an autoencoder by maximizing separation
    between healthy and faulty residuals in training data.
    
    Parameters
    ----------
    errors_train : np.ndarray
        Reconstruction errors for training set.
    y_train : np.ndarray
        Labels for training set (0 = healthy, 1 = faulty)
    num_thresholds : int
        Number of thresholds to try between min and max error.

    Returns
    -------
    best_threshold : float
        Threshold that maximizes TPR - FPR on training data.
    """
    # Range of candidate thresholds
    min_err, max_err = errors_train.min(), errors_train.max()
    thresholds = np.linspace(min_err, max_err, num_thresholds)
    
    best_score = -np.inf
    best_threshold = thresholds[0]

    for t in thresholds:
        # Predicted anomalies above threshold
        preds = errors_train > t

        # True Positive Rate: faulty samples correctly above threshold
        TPR = np.sum((preds == 1) & (y_train == 1)) / max(np.sum(y_train == 1), 1)

        # False Positive Rate: healthy samples incorrectly above threshold
        FPR = np.sum((preds == 1) & (y_train == 0)) / max(np.sum(y_train == 0), 1)

        score = TPR - FPR  # maximal separation

        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold