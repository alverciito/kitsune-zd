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
        "/mnt/8A3A82BA3A82A335/database/CIC2018/02-14-2018.csv",
        "/mnt/8A3A82BA3A82A335/database/CIC2018/02-15-2018.csv",
        "/mnt/8A3A82BA3A82A335/database/CIC2018/02-16-2018.csv",
        # "/mnt/8A3A82BA3A82A335/database/CIC2018/02-20-2018.csv",
        "/mnt/8A3A82BA3A82A335/database/CIC2018/02-21-2018.csv",
        "/mnt/8A3A82BA3A82A335/database/CIC2018/02-22-2018.csv",
        "/mnt/8A3A82BA3A82A335/database/CIC2018/02-23-2018.csv",
        # "/mnt/8A3A82BA3A82A335/database/CIC2018/02-28-2018.csv",
        # "/mnt/8A3A82BA3A82A335/database/CIC2018/03-01-2018.csv",
        "/mnt/8A3A82BA3A82A335/database/CIC2018/03-02-2018.csv"
]



if __name__ == '__main__':
    if not os.path.exists(f'./results'):
        os.makedirs(f'./results')

    try:
        are_categories = list()
        encoder = dict()
        x = None
        y = None
        # Load data:
        for ix, path in enumerate(PATHS):
            (_x, _y), (_encoder, are_categories) = data_loader(path, skip=1_000_000)
            logging.info(f'[DB] {ix}/{len(PATHS)} - {path}')
            if x is None:
                x = _x
                y = _y
                encoder = _encoder
            else:
                x = np.concatenate((x, _x), axis=0)
                y = np.concatenate((y, _y), axis=0)
                for key in _encoder.keys():
                    encoder[key] = _encoder[key]
        logging.info(f'[DATABASE] Loaded {x.shape} rows.')
        x, y = grant_first_label(x, y, 300_000)
        x, y = regularize_data(x, y, are_categories)
        model_experiment_launcher(x, y, f'./results')
    except Exception as e:
        logging.error(f'Error processing CIC2018 database.')
        logging.error(e)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
