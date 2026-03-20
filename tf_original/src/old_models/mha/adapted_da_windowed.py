# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import tensorflow as tf
import numpy as np
import sys
from ..preprocessing import WindowEvaluator, create_windowed_dataset, create_ar_windowed_dataset
EPSILON = sys.float_info.epsilon


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class TransformerDenoisingAutoencoder:
    def __init__(self, n_visible=5, n_hidden=3, lr=0.001, corruption_level=0.0, grace_period=10_000, hidden_ratio=None,
                 name: str = "Anonymous dA", seq_len=500, autoregressive=False):
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
        :param autoregressive: whether the model is autoregressive or not
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
            MultiHeadBlockEncoder(num_heads=self.n_hidden, dim=self.n_visible, encoding_dim=self.n_hidden)
        ])

        # RMSE with SGD:
        self.autoencoder.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.lr), loss='mse')

        # Device:
        self.device = tf.device('/GPU:0') if tf.config.list_physical_devices('GPU') else tf.device('/CPU:0')
        logging.info(f"[{self.name}] Device: {self.device}")

        # Sequence length:
        self.seq_len = seq_len
        self.evaluator = WindowEvaluator(dimensions=self.n_visible, seq_len=seq_len, autoregressive=autoregressive)
        self.create_windowed_dataset = create_ar_windowed_dataset if autoregressive else create_windowed_dataset

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
                x_data = self.create_windowed_dataset(x, window_size=self.seq_len, batch_size=32)
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
            rmse = self.evaluator.evaluate_autoencoder_mse(_x, self.autoencoder)
        return rmse

    def is_in_grace(self):
        return self.epoch < self.grace_period

    def __repr__(self):
        return (f"{self.name}({self.n_visible}, {self.n_hidden}, {self.lr}, {self.corruption_level}, "
                f"{self.grace_period}, {self.hidden_ratio})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class MultiHeadBlockEncoder(tf.keras.Model):
    def __init__(self, num_heads, dim, encoding_dim):
        super(MultiHeadBlockEncoder, self).__init__()
        self.attention_in = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim, value_dim=dim)
        self.dense_0 = tf.keras.layers.Dense(encoding_dim, activation='sigmoid')
        self.dense_1 = tf.keras.layers.Dense(dim, activation='relu')
        self.attention_out = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim, value_dim=dim)

    def call(self, inputs, training=False, mask=None):
        x = self.attention_in(inputs, inputs, inputs)
        x = self.dense_0(x)
        x = self.dense_1(x)
        x = self.attention_out(x, x, x)
        return x
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
