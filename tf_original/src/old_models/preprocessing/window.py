# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import numpy as np
import tensorflow as tf


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

def create_ar_windowed_dataset(data_array, window_size, batch_size):
    """
    Create a windowed autoregressive dataset from a numpy array.
    :param data_array: The original data array
    :param window_size: The size of the input window
    :param batch_size: The batch size
    :return: A tf.data.Dataset object
    """
    data = tf.data.Dataset.from_tensor_slices(data_array)
    # Non-overlapping windows:
    data = data.window(size=window_size, shift=1, drop_remainder=True)
    data = data.flat_map(lambda window: window.batch(window_size))
    # x=y pairs for an autoencoder, optional if you need another type of labels:
    data = data.map(lambda window: (window[..., :-1, :], window[..., -1, :]))
    # Configure the dataset for training:
    data = data.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return data


class WindowEvaluator:
    def __init__(self, dimensions, seq_len=500, autoregressive: bool = False):
        self.seq_len = seq_len
        self.dimensions = dimensions
        self.back_window = None
        self.window_function = create_ar_windowed_dataset if autoregressive else create_windowed_dataset

    def evaluate_autoencoder_mse(self, x, autoencoder):
        """
        Evaluates the autoencoder using MSE.
        :param x: Windowed data tensors
        :param autoencoder: Autoencoder (tf.model)
        :return: A list with the mse.
        """
        # Extend the dataset with self.back_window:
        _x = np.concatenate([self.back_window, x], axis=0)
        dataset = self.window_function(_x, window_size=self.seq_len, batch_size=32)
        predictions = autoencoder.predict(dataset, verbose=1)
        mse = list()

        for idx, (x_batch, _) in enumerate(dataset.unbatch().batch(self.seq_len)):
            z_batch = predictions[idx * self.seq_len:(idx + 1) * self.seq_len, :, :]
            _mse = np.mean(np.mean(np.square(x_batch.numpy() - z_batch), axis=1), axis=1)
            mse.extend(_mse)

        self.back_window = x[-self.seq_len + 1:, :]
        return np.array(mse)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
