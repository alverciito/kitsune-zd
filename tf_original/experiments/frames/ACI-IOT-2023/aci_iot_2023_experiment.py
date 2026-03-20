# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import os
import numpy as np
from experiments.frames.launch_all import model_experiment_launcher, data_loader, regularize_data, grant_first_label
PATHS = [
    '/mnt/8A3A82BA3A82A335/database/ACI-IOT-2023/ACI-IoT-2023.csv'
]
NAMES = [
    'aci_iot_2023'
]



if __name__ == '__main__':
    for name, path in zip(NAMES, PATHS):
        if not os.path.exists(f'./results/{name}'):
            os.makedirs(f'./results/{name}')
        try:
            # Load data:
            (x, y), (encoder, are_categories) = data_loader(path)
            x, y = grant_first_label(x, y, 300_000)
            x, y = regularize_data(x, y, are_categories)
            np.save(f'./target.npy', y)
            model_experiment_launcher(x, y, f'./results/{name}')
        except Exception as e:
            logging.error(f'Error processing database: {name}')
            logging.error(e)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
