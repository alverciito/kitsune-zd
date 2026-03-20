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
    '/mnt/8A3A82BA3A82A335/database/CIC2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    '/mnt/8A3A82BA3A82A335/database/CIC2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    '/mnt/8A3A82BA3A82A335/database/CIC2017/Friday-WorkingHours-Morning.pcap_ISCX.csv',
    '/mnt/8A3A82BA3A82A335/database/CIC2017/Monday-WorkingHours.pcap_ISCX.csv',
    '/mnt/8A3A82BA3A82A335/database/CIC2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    '/mnt/8A3A82BA3A82A335/database/CIC2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    '/mnt/8A3A82BA3A82A335/database/CIC2017/Tuesday-WorkingHours.pcap_ISCX.csv',
    '/mnt/8A3A82BA3A82A335/database/CIC2017/Wednesday-workingHours.pcap_ISCX.csv'
]
NAMES = [
    'cic2017_friday_ddos',
    'cic2017_friday_portscan',
    'cic2017_friday_morning',
    'cic2017_monday',
    'cic2017_thursday_infilteration',
    'cic2017_thursday_webattacks',
    'cic2017_tuesday',
    'cic2017_wednesday'
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
        for name, path in zip(NAMES, PATHS):
            (_x, _y), (_encoder, are_categories) = data_loader(path)
            if x is None:
                x = _x
                y = _y
                encoder = _encoder
            else:
                x = np.concatenate((x, _x), axis=0)
                y = np.concatenate((y, _y), axis=0)
                for key in _encoder.keys():
                    encoder[key] = _encoder[key]

        x, y = grant_first_label(x, y, 300_000)
        x, y = regularize_data(x, y, are_categories)
        model_experiment_launcher(x, y, f'./results')
    except Exception as e:
        logging.error(f'Error processing CIC2017 database.')
        logging.error(e)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
