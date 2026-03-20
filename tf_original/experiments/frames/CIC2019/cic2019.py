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
    ["/mnt/8A3A82BA3A82A335/database/CIC2019/DrDoS_DNS_data_1_per.csv"],
    ["/mnt/8A3A82BA3A82A335/database/CIC2019/DrDoS_LDAP_data_2_0_per.csv"],
    ["/mnt/8A3A82BA3A82A335/database/CIC2019/DrDoS_MSSQL_data_1_3_per.csv"],
    ["/mnt/8A3A82BA3A82A335/database/CIC2019/DrDoS_NetBIOS_data_1_3_per.csv"],
    ["/mnt/8A3A82BA3A82A335/database/CIC2019/DrDoS_NTP_data_data_5_per.csv"],
    ["/mnt/8A3A82BA3A82A335/database/CIC2019/DrDoS_SNMP_data_1_3_per.csv"],
    ["/mnt/8A3A82BA3A82A335/database/CIC2019/DrDoS_SSDP_data_2_per.csv"],
    ["/mnt/8A3A82BA3A82A335/database/CIC2019/DrDoS_UDP_data_2_per.csv"]
]

if __name__ == '__main__':
    try:
        for index, PATH in enumerate(PATHS):

            if not os.path.exists(f'./results/{index}'):
                os.makedirs(f'./results/{index}')

            are_categories = list()
            encoder = dict()
            x = None
            y = None
            # Load data:
            for ix, path in enumerate(PATH):
                (_x, _y), (_encoder, are_categories) = data_loader(path)
                logging.info(f'[DB] {ix}/{len(PATH)} - {path}')
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
            model_experiment_launcher(x, y, f'./results/{index}')
    except Exception as e:
        logging.error(f'Error processing CIC2019 database.')
        logging.error(e)
        raise e
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
