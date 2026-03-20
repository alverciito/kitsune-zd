# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import tqdm
from src.models import KitNET, NETWORK_TYPES
from src.database import TestDatabase
import logging
import tensorflow as tf
FEATURE_SPACE = 115
TEST_LEN = 400_000_000_000
TRAIN_LEN = 200_000



# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    # Create a new KitNET instance and database.
    db = TestDatabase(regularized=False)
    logging.info(f'Network types: {NETWORK_TYPES}')

    with tf.device('/device:GPU:1'):
        for db_idx, (x, y) in enumerate(db):

            logging.info(f'Processing database: {db.db_names[db_idx]}')
            # Format y:
            if len(x) != len(y):
                y = y[:len(x)]
            y = y[:TEST_LEN]
            x = x[:TEST_LEN]

            for model_type in NETWORK_TYPES:
                if model_type == 'stat':
                    logging.info(f'Processing model: {model_type}')

                    kn = KitNET(FEATURE_SPACE,
                                feature_map=None,
                                    ad_grace_period=150_000,
                                    execution_window=300_000,
                                    fm_grace_period=50_000,
                                    sequence_length=1_000,
                                    hidden_ratio=0.2,
                                    autoencoder_size=4,
                                    clustering='dbscan',
                                    ar=True,
                                    ae_type=model_type,
                                    output_ae_type='stat')

                    # Process packet with KitNet:
                    with tqdm.tqdm(total=(len(x)), desc='Processing data') as pbar:
                        for _, packet in enumerate(x):
                            kn.process(packet, is_last=False if _ < len(x) - 1 else True)
                            pbar.update(1)
                    # Save the rmse values:
                    kn.show_stats(y)
                    del kn
            break
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
