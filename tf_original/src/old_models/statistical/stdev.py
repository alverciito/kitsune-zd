# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import numpy as np
import matplotlib.pyplot as plt


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class StdAnomaly:
    def __init__(self, n_features, grace_period=200_000, execution_window=500_000):
        """
        This class implements a simple anomaly detector based on the standard deviation of the data.
        :param n_features: The number of features.
        :param grace_period: The number of samples to wait before training.
        :param execution_window: The number of samples to wait before executing.
        """
        # Properties:
        self.buffer = list()
        self.n_features = n_features
        self.grace_period = grace_period
        self.execution_window = execution_window

        # Train variables:
        self.mean = None
        self.stdev = None
        self.norm_max = None
        self.norm_min = None

        # Error storage:
        self.error = list()
        self.current_error = None
        self.epoch = 0

    def process(self, x, is_last=False):
        """
        Process the data.
        :param x: The data.
        :param is_last: If True, the last data point is being processed.
        :return: The error deviation.
        """
        x_reshaped = x.reshape((-1, self.n_features))
        self.buffer.append(x_reshaped)
        self.epoch += len(x_reshaped)
        if self.epoch >= self.grace_period and self.stdev is None:
            error = self.train()
            self.error.append(error)
            self.buffer = list()
            self.epoch = 0
        if (self.epoch >= self.execution_window and self.stdev is not None) or is_last:
            error = self.execute()
            self.error.append(error)
            self.buffer = list()
            self.epoch = 0
        return self.error


    def train(self):
        """
        Train the model using the buffer.
        :return: The error deviation.
        """
        # Normalize the data
        buffer = np.concatenate(self.buffer, axis=0)
        self.norm_max = np.max(buffer, axis=0)
        self.norm_min = np.min(buffer, axis=0)
        norm = (buffer - self.norm_min) / (self.norm_max - self.norm_min + 1e-16)

        # Update the mean and the standard deviation
        self.mean = np.mean(norm, axis=0)
        self.stdev = np.std(norm, axis=0)

        # Compute the error:
        dev = (norm - self.mean) / (self.stdev + 1e-16)
        error = np.mean(np.power(dev, 2), axis=1)
        return error


    def execute(self):
        """
        Execute the model using the buffer.
        :return: The error deviation.
        """
        # Normalize the data
        buffer = np.concatenate(self.buffer, axis=0)
        norm = (buffer - self.norm_min) / (self.norm_max - self.norm_min + 1e-16)

        # Compute the error:
        dev = (norm - self.mean) / (self.stdev + 1e-16)
        error = np.mean(np.power(dev, 2), axis=1)
        return error

    def show_stats(self, y: np.ndarray = None, save: str = None):
        """
        Show the statistics of the model.
        :param y: The labels of the dataset if provided.
        :param save: The path to save the plot.
        :return: Nothing.
        """
        self.current_error = np.concatenate(self.error, axis=0)
        mse = np.array(self.current_error)
        mse = np.log10(mse + 1)
        grace_period_ad = self.grace_period
        # Plot the MSE:
        plt.figure(figsize=(10, 5))
        if y is not None:
            plt.fill_between(np.arange(len(mse)), 0, max(mse) * y, color='gray', alpha=0.1, label='Anomaly')
        plt.plot(mse, label="RMSE", alpha=0.8, linewidth=0., color='black', marker='o', markersize=1.)
        plt.axvline(grace_period_ad, color='black', alpha=0.5, linestyle='--', label='Stop AD training')
        plt.xlim(0, len(mse))
        plt.ylim(0, max(mse) * 1.01)
        # yticks in log scale:
        plt.title(f'Log error for "STDev" model')
        plt.xlabel('Packet Number')
        plt.ylabel('$\log_{10}({RMSE})$')
        if save is not None:
            plt.savefig(save)
        plt.show()

    @property
    def current_rmse(self):
        return  np.concatenate(self.error, axis=0)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
