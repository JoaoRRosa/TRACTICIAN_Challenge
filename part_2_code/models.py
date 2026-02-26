from tensorflow.keras import layers, models

# ---------------------------
# 2D CNN Autoencoder
# ---------------------------
def build_2d_cnn(input_shape, filters, optimizer="adam"):

    import tensorflow as tf
    layers = tf.keras.layers
    models = tf.keras.models

    if not isinstance(filters, (list, tuple)):
        raise ValueError("filters must be a list like [32, 16]")

    inp = layers.Input(shape=input_shape)

    # -------- Encoder --------
    x = inp
    for f in filters:
        x = layers.Conv2D(f, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 1), padding='same')(x)

    # -------- Decoder --------
    for f in reversed(filters):
        x = layers.Conv2DTranspose(
            f,
            3,
            strides=(2, 1),
            activation='relu',
            padding='same'
        )(x)

    decoded = layers.Conv2D(
        input_shape[-1],
        3,
        activation='linear',
        padding='same'
    )(x)

    model = models.Model(inp, decoded)
    model.compile(optimizer=optimizer, loss='mse')

    return model
# ---------------------------
# 1D CNN Autoencoder
# ---------------------------
def build_1d_cnn(input_shape, filters, optimizer="adam"):

    inp = layers.Input(shape=input_shape)
    x = inp

    # -------- Encoder --------
    for f in filters:
        x = layers.Conv1D(f, 3, activation="relu", padding="same")(x)
        x = layers.MaxPooling1D(2, padding="same")(x)

    # -------- Decoder --------
    for f in reversed(filters):
        x = layers.Conv1DTranspose(
            f,
            3,
            strides=2,
            activation="relu",
            padding="same"
        )(x)

    x = layers.Conv1D(
        input_shape[-1],
        3,
        activation="linear",
        padding="same"
    )(x)

    # -------- FORCE MATCH LENGTH --------
    output_len = x.shape[1]
    input_len = input_shape[0]

    if output_len != input_len:
        crop = output_len - input_len
        x = layers.Cropping1D((0, crop))(x)

    model = models.Model(inp, x)
    model.compile(optimizer=optimizer, loss="mse")

    return model

# ---------------------------
# LSTM Autoencoder
# ---------------------------
def build_lstm_ae(input_shape, latent_dim, optimizer):

    inp = layers.Input(shape=input_shape)

    encoded = layers.LSTM(latent_dim, return_sequences=False)(inp)
    x = layers.RepeatVector(input_shape[0])(encoded)
    x = layers.LSTM(latent_dim, return_sequences=True)(x)

    decoded = layers.TimeDistributed(
        layers.Dense(input_shape[1])
    )(x)

    model = models.Model(inp, decoded)
    model.compile(optimizer=optimizer, loss="mse")

    return model