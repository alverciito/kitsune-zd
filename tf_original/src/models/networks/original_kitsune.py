# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import numpy as np
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class DenoisingAutoencoder:
    def __init__(self, n_visible=5, n_hidden=3, lr=0.001, corruption_level=0.0, grace_period=10_000, hidden_ratio=None,
                 seed=1234, **kwargs):
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
        self.n_visible = n_visible
        self.lr = lr
        self.corruption_level = corruption_level
        self.grace_period = grace_period
        self.hidden_ratio = hidden_ratio
        if hidden_ratio is not None:
            self.n_hidden = int(np.ceil(self.n_visible * self.hidden_ratio))
        else:
            self.n_hidden = n_hidden

        # Normalization parameters
        self.norm_max = np.ones((self.n_visible,)) * -np.Inf
        self.norm_min = np.ones((self.n_visible,)) * np.Inf
        self.n = 0

        # Random number generator
        self.rng = np.random.RandomState(seed)

        # Initialize weights
        a = 1. / self.n_visible
        self.W = np.array(self.rng.uniform(low=-a, high=a, size=(self.n_visible, self.n_hidden)))
        self.h_bias = np.zeros(self.n_hidden)   # initialize h bias 0
        self.v_bias = np.zeros(self.n_visible)  # initialize v bias 0
        self.W_prime = self.W.T                 # transpose of W

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

    def train(self, input_x):
        """
        Trains the dA model.
        :param input_x: The input data
        :return: The RMSE reconstruction error during training.
        """
        l_h2_list = list()
        for x in input_x:
            # update norms
            self.norm_max[x > self.norm_max] = x[x > self.norm_max]
            self.norm_min[x < self.norm_min] = x[x < self.norm_min]

            # 0-1 normalize
            x = (x - self.norm_min) / (self.norm_max - self.norm_min + 1e-6)

            if self.corruption_level > 0.0:
                tilde_x = self.dropout(x, self.corruption_level)
            else:
                tilde_x = x
            y = self.forward_encode(tilde_x)
            z = self.forward_decode(y)

            L_h2 = x - z
            L_h1 = np.dot(L_h2, self.W) * y * (1 - y)

            L_vbias = L_h2
            L_hbias = L_h1
            L_W = np.outer(tilde_x.T, L_h1) + np.outer(L_h2.T, y)

            self.W += self.lr * L_W
            self.h_bias += self.lr * L_hbias
            self.v_bias += self.lr * L_vbias
            l_h2_list.append(L_h2)

        self.n += len(input_x)
        # Compute last RMSE
        l_h2_array = np.array(l_h2_list)
        return np.sqrt(np.mean(l_h2_array**2, axis=1)) #the RMSE reconstruction error during training


    def reconstruct(self, x):
        y = self.forward_encode(x)
        z = self.forward_decode(y)
        return z

    def execute(self, x): #returns MSE of the reconstruction of x
        if self.n < self.grace_period:
            return 0.0
        else:
            # 0-1 normalize
            x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)
            z = self.reconstruct(x)
            rmse = np.sqrt(np.mean((x - z) ** 2, axis=1)) #MSE
            return rmse

    def inGrace(self):
        return self.n < self.grace_period
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
