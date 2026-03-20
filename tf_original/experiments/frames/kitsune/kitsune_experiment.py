# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import os
from experiments.frames.launch_all import model_experiment_launcher
from src.database import TestDatabase


if __name__ == '__main__':
    # Create a new KitNET instance and database.
    db = TestDatabase(regularized=False)

    for db_idx, (x, y) in enumerate(db):
        logging.info(f'Processing database: {db.db_names[db_idx]}')
        if not os.path.exists(f'./results/{db.db_names[db_idx]}'):
            os.makedirs(f'./results/{db.db_names[db_idx]}')
        try:
            model_experiment_launcher(x, y, f'./results/{db.db_names[db_idx]}')
        except Exception as e:
            logging.error(f'Error processing database: {db.db_names[db_idx]}')
            logging.error(e)
            continue
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
