# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import numpy as np
from sklearn.cluster import KMeans as skKMeans

class KMeans:
    def __init__(self, n):
        #parameter:
        self.n = n
        #varaibles
        self.buffer = list()

    # x: a numpy vector of length n
    def update(self,x):
        self.buffer.append(x)

    # clusters the features together, having no more than maxClust features per cluster
    def cluster(self, n_clusters):
        x = np.array(self.buffer)
        self.buffer = list()
        kmeans = skKMeans(n_clusters=n_clusters, random_state=0).fit(x.T)
        sk_labels = kmeans.labels_

        feature_map = list()
        for i in range(n_clusters):
            feature_map.append(np.where(sk_labels == i)[0].tolist())
        return feature_map
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
