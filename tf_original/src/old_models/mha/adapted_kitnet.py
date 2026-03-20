# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import numpy as np
from .adapted_da_windowed import TransformerDenoisingAutoencoder
from .adapted_da_windowed_or import DenoisingAutoencoder
from .adapted_corclust import corClust
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #


class KitNET:
    def __init__(self, n, max_autoencoder_size=10, sequence_length=500,
                 fm_grace_period=None, ad_grace_period=10_000, execution_window=10_000,
                 learning_rate=0.1, hidden_ratio=0.75, feature_map=None):
        """
        KitNET constructor.
        :param n: The input dimension space (number of columns in data).
        :param max_autoencoder_size: The maximum size of the autoencoders.
        :param fm_grace_period: The number of instances to train the feature mapper before execute mode.
        :param ad_grace_period: The number of instances to train the model before execute mode.
        :param learning_rate: The default SGD learning rate for all autoencoders in the KitNET instance.
        :param hidden_ratio: Compression factor used to determine the number of nodes in the autoencoder hidden layer.
        :param feature_map: One may optionally provide a feature map instead of learning one. The map must be a list,
                where the i-th entry contains a list of the feature indices to be assigned to the i-th autoencoder in
                the ensemble. For example, [[2,5,3],[4,0,1],[6,7]]
        :param execution_window: The number of instances to process before updating the ensemble layer.
        :param sequence_length: The length of the input window.
        """
        # Parameters:
        self.ad_grace_period = ad_grace_period
        if fm_grace_period is None:
            self.fm_grace_period = ad_grace_period
        else:
            self.fm_grace_period = fm_grace_period
        if max_autoencoder_size <= 0:
            self.m = 1
        else:
            self.m = max_autoencoder_size
        self.lr = learning_rate
        self.hr = hidden_ratio
        self.n = n

        # Variables
        self.n_trained = 0      # The number of training instances so far.
        self.v = feature_map
        if feature_map is None:
            logging.info("[Feature-Mapper]: train-mode, Anomaly-Detector: off-mode")
        else:
            self.__createAD__()
            logging.info("[Feature-Mapper]: execute-mode, Anomaly-Detector: train-mode")
        self.fm = corClust(self.n)   # Incremental feature clustering for the feature mapping process.
        self.ensemble_layer = list()
        self.output_layer = None

        # Execution window:
        self.execution_window_size = execution_window
        self.execution_window = list()
        self.current_rmse = list()
        # Sequence len:
        self.seq_len = sequence_length

    def process(self, x: np.ndarray, is_last=False):
        """
        Learning or execute mode of KitNET.
        :param x: A numpy array of length n.
        :param is_last: If True, the last instance of the dataset is being processed.
        :return: The anomaly score of x during training (do not use for alerting).
        """
        if self.n_trained > self.fm_grace_period + self.ad_grace_period:  # If both the FM and AD are in execute-mode.
            self.execute(x, is_last)
        else:
            self.train(x)
            self.current_rmse.append(0.0)

    def train(self, x: np.ndarray):
        """
        Force train KitNET on x.
        :param x: A numpy array of length n.
        :return: Nothing.
        """
        # If the FM is in train-mode, and the user has not supplied a feature mapping
        if self.n_trained <= self.fm_grace_period and self.v is None:
            # Update the incremental correlation matrix
            self.fm.update(x)
            if self.n_trained == self.fm_grace_period:  # If the feature mapping should be instantiated
                self.v = self.fm.cluster(self.m)
                # self.v.sort(key=lambda _: len(_), reverse=True)  # TODO: Remove sort
                self.__createAD__()
                logging.info("The Feature-Mapper found a mapping: "+str(self.n)+" features to "
                                                                                ""+str(len(self.v))+" autoencoders.")
                logging.info("Feature-Mapper: execute-mode, Anomaly-Detector: train-mode")
        else:
            # Ensemble layer.
            s_l1 = list()
            for a, ensemble in enumerate(self.ensemble_layer):
                # Make sub instance for autoencoder 'a'.
                xi = x[self.v[a]]
                s_l1.append(ensemble.train(xi))

            # # Output layer.
            # self.output_layer.train(s_l1)
            if self.n_trained == self.ad_grace_period + self.fm_grace_period:
                # Output layer:
                try:
                    s_l1 = np.array(s_l1)
                    self.output_layer.grace_period = s_l1.shape[-1]
                    self.output_layer.epoch = self.output_layer.grace_period - 1
                    self.output_layer.train(s_l1.T)
                    logging.info("Feature-Mapper: execute-mode, Anomaly-Detector: execute-mode")
                except Exception as e:
                    logging.error(f"Error: {e}")
        self.n_trained += 1

    def execute(self, x, is_last=False):
        if self.v is None:
            raise RuntimeError('KitNET Cannot execute x, because a feature mapping has not yet been learned '
                               'or provided. Try running process(x) instead.')
        self.execution_window.append(x)
        if len(self.execution_window) >= self.execution_window_size or is_last:
            # Ensemble Layer
            _x = np.array(self.execution_window)
            s_l1 = list()
            for a, ensemble in enumerate(self.ensemble_layer):
                # make sub inst
                xi = _x[:, self.v[a]]
                s_l1.append(ensemble.execute(xi))
            # Output layer
            output = self.output_layer.execute(np.array(s_l1).T)
            self.current_rmse.extend(output)
            self.execution_window = list()
        # Output layer.
        return

    def __createAD__(self):
        # Construct ensemble layer
        for ix, fmap in enumerate(self.v):
            self.ensemble_layer.append(TransformerDenoisingAutoencoder(
                n_visible=len(fmap),
                n_hidden=0,
                lr=self.lr,
                corruption_level=0,
                grace_period=self.ad_grace_period,
                hidden_ratio=self.hr,
                name=f'Autoencoder {ix}',
                seq_len=self.seq_len))

        # construct output layer
        self.output_layer = DenoisingAutoencoder(len(self.v),
                                                 n_hidden=0,
                                                 lr=self.lr,
                                                 corruption_level=0,
                                                 grace_period=self.ad_grace_period,
                                                 hidden_ratio=self.hr,
                                                 name='Output Layer')
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
