# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from kitsunezd.__special__ import logging
import tensorflow as tf
import numpy as np
import sys
EPSILON = sys.float_info.epsilon


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class TransformerDenoisingAutoencoder:
    def __init__(self, n_visible=5, n_hidden=3, lr=0.001, corruption_level=0.0, grace_period=10_000, hidden_ratio=None,
                 name: str = "Anonymous dA", seq_len=500):
        """
        Denoising Autoencoder (dA) class.
        :param n_visible: number of units in visible (input) layer
        :param n_hidden: number of units in hidden layer
        :param lr: learning rate
        :param corruption_level: drop-out probability
        :param grace_period: number of samples to observe before training
        :param hidden_ratio: ratio of hidden units to visible units
        :param name: name of the model
        :param seq_len: sequence length of the input window
        """
        self.name = name
        self.n_visible = n_visible
        self.lr = lr
        self.corruption_level = corruption_level
        self.grace_period = grace_period
        self.hidden_ratio = hidden_ratio
        self.grace_window = list()
        if hidden_ratio is not None:
            self.n_hidden = int(np.ceil(self.n_visible * self.hidden_ratio))
        else:
            self.n_hidden = n_hidden

        # Normalization parameters
        self.norm_max = np.ones((self.n_visible,)) * -np.Inf
        self.norm_min = np.ones((self.n_visible,)) * np.Inf
        self.epoch = 0

        # Keras autoencoder:
        self.autoencoder = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(seq_len, self.n_visible)),
            tf.keras.layers.Conv1D(3, seq_len, padding='same', activation='relu'),
            tf.keras.layers.Dense(self.n_hidden, activation='relu'),
            tf.keras.layers.Dense(self.n_visible, activation='relu')
        ])

        # RMSE with SGD:
        self.autoencoder.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.lr), loss='mse')

        # Device:
        self.device = tf.device('/GPU:0') if tf.config.list_physical_devices('GPU') else tf.device('/CPU:0')
        logging.info(f"[{self.name}] Device: {self.device}")

        # Sequence length:
        self.seq_len = seq_len
        self.evaluator = Evaluator(dimensions=self.n_visible, seq_len=seq_len)

    def train(self, x):
        """
        Trains the dA model.
        :param x: The input data
        :return: The RMSE reconstruction error during training.
        """
        # Update epoch:
        self.epoch = self.epoch + 1
        rmse = list()

        if self.epoch == self.grace_period:
            # Convert to np.array:
            self.grace_window.append(x)
            grace_window = np.array(self.grace_window).reshape(self.grace_period, -1)
            # Update norms and normalize:
            self.norm_max = np.max(grace_window, axis=0)
            self.norm_min = np.min(grace_window, axis=0)
            x = (grace_window - self.norm_min) / (self.norm_max - self.norm_min + EPSILON)

            # Train the model:
            logging.info(f"Training {self.name} model with {self.grace_period} samples.")
            with self.device:
                x_data = create_windowed_dataset(x, window_size=self.seq_len, batch_size=32)
                history = self.autoencoder.fit(x_data, epochs=1, verbose=1)
                rmse = history.history['loss']

            self.evaluator.back_window = x[-self.seq_len + 1:, :]

        elif self.epoch < self.grace_period:
            self.grace_window.append(x)
            rmse.append(0.0)
        else:
            raise RuntimeError("The dA model is in execute mode and cannot be trained.")
        return np.array(rmse)

    def execute(self, x: np.ndarray):
        """
        Executes the dA model.
        :param x: The input data (batch, n_visible)
        :return: The RMSE reconstruction error during execution.
        """
        if self.is_in_grace():
            rmse = [-0.0] * x.shape[0]
        else:
            # Append:
            _x = x.reshape(-1, self.n_visible)
            # Normalize:
            _x = (_x - self.norm_min) / (self.norm_max - self.norm_min + EPSILON)
            # Dataset:
            # x_data = create_windowed_dataset(_x, window_size=self.seq_len, batch_size=32)
            # z = self.autoencoder.predict(x_data, verbose=1)
            # rmse = np.sqrt(((_x - z) ** 2).mean(axis=1)).tolist()
            rmse = self.evaluator.evaluate_autoencoder_mse(_x, self.autoencoder)
            # print(f"Input: {x.shape}, {_x.shape} Output: {len(rmse)}")
        return rmse

    def is_in_grace(self):
        return self.epoch < self.grace_period

    def __repr__(self):
        return (f"{self.name}({self.n_visible}, {self.n_hidden}, {self.lr}, {self.corruption_level}, "
                f"{self.grace_period}, {self.hidden_ratio})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def create_windowed_dataset(data_array, window_size, batch_size):
    """
    Create a windowed dataset from a numpy array.
    :param data_array: The original data array
    :param window_size: The size of the window
    :param batch_size: The batch size
    :return: A tf.data.Dataset object
    """
    data = tf.data.Dataset.from_tensor_slices(data_array)
    # Non-overlapping windows:
    data = data.window(size=window_size, shift=1, drop_remainder=True)
    data = data.flat_map(lambda window: window.batch(window_size))
    # x=y pairs for an autoencoder, optional if you need another type of labels:
    data = data.map(lambda window: (window, window))
    # Configure the dataset for training:
    data = data.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return data


class Evaluator:
    def __init__(self, dimensions, seq_len=500):
        self.seq_len = seq_len
        self.dimensions = dimensions
        self.back_window = None
        # self.back_window = np.zeros((self.seq_len - 1, dimensions))

    def evaluate_autoencoder_mse(self, x, autoencoder):
        """
        Evalúa el autoencoder utilizando un tf.data.Dataset y calcula el MSE para cada ejemplo.

        :param x: tensor que contiene las ventanas de datos
        :param autoencoder: Modelo de autoencoder entrenado
        :return: Lista de MSE para cada ventana en el dataset
        """
        # Extend the dataset with self.back_window:
        _x = np.concatenate([self.back_window, x], axis=0)
        dataset = create_windowed_dataset(_x, window_size=self.seq_len, batch_size=32)
        predictions = autoencoder.predict(dataset, verbose=1)
        mse = []

        for idx, (x_batch, _) in enumerate(dataset.unbatch().batch(self.seq_len)):
            z_batch = predictions[idx * self.seq_len:(idx + 1) * self.seq_len, :, :]
            _mse = np.mean(np.mean(np.square(x_batch.numpy() - z_batch), axis=1), axis=1)
            mse.extend(_mse)

        self.back_window = x[-self.seq_len + 1:, :]
        return np.array(mse)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
