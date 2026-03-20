# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import numpy as np
import sys
from kitsunezd.addos.utils import *

EPSILON = sys.float_info.epsilon


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class DenoisingAutoencoder:
    def __init__(self, n_visible=5, n_hidden=3, lr=0.001, corruption_level=0.0, grace_period=10_000, hidden_ratio=None,
                 seed=1234, name: str = "Anonymous dA"):
        """
        Denoising Autoencoder (dA) class.
        :param n_visible: number of units in visible (input) layer
        :param n_hidden: number of units in hidden layer
        :param lr: learning rate
        :param corruption_level: drop-out probability
        :param grace_period: number of samples to observe before training
        :param hidden_ratio: ratio of hidden units to visible units
        :param seed: random seed
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

        # Random number generator
        self.rng = np.random.RandomState(seed)

        # Initialize weights
        a = 1. / self.n_visible
        self.W = np.array(self.rng.uniform(low=-a, high=a, size=(self.n_visible, self.n_hidden)))
        self.h_bias = np.zeros(self.n_hidden)  # initialize h bias 0
        self.v_bias = np.zeros(self.n_visible)  # initialize v bias 0
        self.W_prime = self.W.T  # transpose of W

    def dropout(self, x, dropout_rate):
        """
        Applies dropout to data.
        :param x: The input data
        :param dropout_rate: The drop-out probability
        :return: The drop-out input
        """
        if dropout_rate >= 1:
            raise ValueError("Corruption level must be in the range [0, 1].")
        return self.rng.binomial(size=x.shape, n=1, p=1 - dropout_rate) * x

    def forward_encode(self, x):
        """
        Encodes the input data.
        :param x: The input data
        :return: The encoded data
        """
        return sigmoid(np.dot(x, self.W) + self.h_bias)

    def forward_decode(self, hidden):
        """
        Decodes the hidden data.
        :param hidden: The hidden data.
        :return: The decoded data.
        """
        return sigmoid(np.dot(hidden, self.W_prime) + self.v_bias)

    def forward(self, x):
        """
        Forward pass of the dA model.
        :param x: The input data
        :return: The reconstructed data
        """
        y = self.forward_encode(x)
        z = self.forward_decode(y)
        return z

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

            for _ in x:
                # Apply dropout:
                if self.corruption_level > 0.0:
                    tilde_x = self.dropout(_, self.corruption_level)
                else:
                    tilde_x = _

                # Forward pass:
                y = self.forward_encode(tilde_x)
                z = self.forward_decode(y)

                # Compute gradients:
                l_h2 = tilde_x - z
                l_h1 = np.dot(l_h2, self.W) * y * (1 - y)
                l_vbias = l_h2
                l_hbias = l_h1
                _outer1 = np.outer(tilde_x.T, l_h1)
                _outer2 = np.outer(l_h2.T, y)
                l_w = _outer1 + _outer2

                # Check nan:
                if np.isnan(l_w).any() or np.isnan(l_hbias).any() or np.isnan(l_vbias).any():
                    raise RuntimeError("NaN detected in the gradients.")

                # Update weights:
                self.W += self.lr * l_w
                self.h_bias += self.lr * l_hbias
                self.v_bias += self.lr * l_vbias
                rmse.append(np.sqrt(np.mean(l_h2 ** 2)))
        elif self.epoch < self.grace_period:
            self.grace_window.append(x)
            rmse.append(0.0)
        else:
            raise RuntimeError("The dA model is in execute mode and cannot be trained.")
        return np.array(rmse)

    def execute(self, x):
        if self.is_in_grace():
            rmse = 0.0
        else:
            # Normalize:
            x = (x - self.norm_min) / (self.norm_max - self.norm_min + EPSILON)
            z = self.forward(x)
            rmse = np.sqrt(((x - z) ** 2).mean())
        return rmse

    def is_in_grace(self):
        return self.epoch < self.grace_period

    def __repr__(self):
        return (f"{self.name}({self.n_visible}, {self.n_hidden}, {self.lr}, {self.corruption_level}, "
                f"{self.grace_period}, {self.hidden_ratio})")
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
