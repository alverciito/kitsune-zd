# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import numpy as np

def create_feature_map(n, clusters):
    # clusters is the number of clusters and n is the number of features
    feature_map = []
    available_choices = np.arange(n)
    shuffled = np.random.permutation(available_choices)
    for i in range(clusters):
        feature_map.append(shuffled[i::clusters])
    return feature_map
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
