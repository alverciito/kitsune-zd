# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
EPSILON = sys.float_info.epsilon
# CUDA setup:
if torch.cuda.is_available():
    print("GPUs disponibles:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print("GPU", i, ":", torch.cuda.get_device_name(i))
else:
    print("No hay GPUs disponibles, utilizando CPU.")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class DenoisingAutoencoder(nn.Module):
    def __init__(self, n_visible=5, n_hidden=3, lr=0.001, corruption_level=0.0, grace_period=10_000, hidden_ratio=None,
                 seed=1234, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
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
        # Configure model to use multiple GPUs
        super(DenoisingAutoencoder, self).__init__()
        self.device = device

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
        self.norm_max = torch.from_numpy(np.ones((self.n_visible,)) * -np.Inf).float().to(device)
        self.norm_min = torch.from_numpy(np.ones((self.n_visible,)) * np.Inf).float().to(device)
        self.epoch = 0

        # Random number generator
        torch.manual_seed(seed)

        # Convert scalar to tensor for operations
        scaling_factor = torch.sqrt(torch.tensor(2. / n_visible))
        # Initialize weights with scaling
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * scaling_factor)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))

        # Setup optimizer
        self.dropout = nn.Dropout(p=self.corruption_level)

        # Optimizer
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)

        # Device
        self.to(device)
        print("Denoising Autoencoder model created.")

    def forward_encode(self, x):
        """
        Encodes the input data.
        :param x: The input data
        :return: The encoded data
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        return torch.sigmoid(torch.matmul(x, self.W) + self.h_bias)

    def forward_decode(self, x):
        """
        Decodes the hidden data.
        :param x: The hidden data.
        :return: The decoded data.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        return torch.sigmoid(torch.matmul(x, self.W.t()) + self.v_bias)

    def forward(self, x):
        """
        Forward pass of the dA model.
        :param x: The input data
        :return: The reconstructed data
        """
        y = self.forward_encode(x)
        z = self.forward_decode(y)
        return z

    def train(self, x, mode=False):
        """
        Trains the dA model.
        :param x: The input data
        :return: The RMSE reconstruction error during training.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(np.array(x)).float()
        x = x.to(self.device)

        # Update norms and normalize:
        self.norm_max[x > self.norm_max] = x[x > self.norm_max]
        self.norm_min[x < self.norm_min] = x[x < self.norm_min]

        self.optimizer.zero_grad()  # Clear gradients
        # Normalize input
        x_normalized = (x - self.norm_min) / (self.norm_max - self.norm_min + EPSILON)
        # Apply dropout
        x_corrupted = self.dropout(x_normalized)
        # Forward pass
        x_reconstructed = self.forward(x_corrupted)
        # Compute loss
        loss = torch.sqrt(((x_normalized - x_reconstructed) ** 2).mean())
        # Backward pass
        loss.backward()
        self.optimizer.step()
        # Update epoch:
        self.epoch = self.epoch + 1
        return loss.item()

    def execute(self, x):
        self.eval()
        with torch.no_grad():
            if self.epoch < self.grace_period:
                rmse = 0.0
            else:
                # Normalize:
                x_normalized = (x - self.norm_min) / (self.norm_max - self.norm_min + EPSILON)
                x_reconstructed = self.forward(x_normalized)
                rmse = torch.sqrt(((x_normalized - x_reconstructed) ** 2).mean())
            return rmse.item()

    def is_in_grace(self):
        return self.epoch < self.grace_period
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
