# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from src.old_models.mha import MHAKitNet
from src.old_models.conv1d import Conv1DKitNet
from src.old_models.kitnet import KitNET
from src.old_models.statistical import StdAnomaly
from src.database import TestDatabase
import logging
FEATURE_SPACE = 115
TEST_LEN = 400_000_000_000
TRAIN_LEN = 200_000

def create_kitnet():
    return KitNET(FEATURE_SPACE, max_autoencoder_size=10, execution_window=200_000,
    fm_grace_period=TRAIN_LEN // 2,
    ad_grace_period=TRAIN_LEN // 2,
    learning_rate=0.1, hidden_ratio=0.75, feature_map=None)

def create_mhakitnet():
    return MHAKitNet(FEATURE_SPACE, max_autoencoder_size=10, execution_window=200_000,
    fm_grace_period=TRAIN_LEN // 2,
    ad_grace_period=TRAIN_LEN // 2,
    learning_rate=0.1, hidden_ratio=0.75, feature_map=None, sequence_length=800)

def create_statisticalkitnet():
    return StdAnomaly(FEATURE_SPACE, grace_period=TRAIN_LEN // 2, execution_window=200_000)

def create_conv1dkitnet():
    return Conv1DKitNet(FEATURE_SPACE, max_autoencoder_size=10, execution_window=200_000,
    fm_grace_period=TRAIN_LEN // 2,
    ad_grace_period=TRAIN_LEN // 2,
    learning_rate=0.1, hidden_ratio=0.75, feature_map=None, sequence_length=800)

def create_moekitnet():
    return None


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    # Create a new KitNET instance and database.
    db = TestDatabase(regularized=False)

    for db_idx, (x, y) in enumerate(db):
        kn = create_conv1dkitnet()
        # Format y:
        if len(x) != len(y):
            y = y[:len(x)]
        y = y[:TEST_LEN]
        x = x[:TEST_LEN]

        logging.info(f'Processing database: {db.db_names[db_idx]}')
        # Process packet with KitNet:
        with tqdm.tqdm(total=(len(x)), desc='Processing data') as pbar:
            for _, packet in enumerate(x):
                kn.process(packet, is_last=False if _ < len(x) - 1 else True)
                pbar.update(1)
        # Save the rmse values:
        rmse = np.array(kn.current_rmse)

        # Scale RMSE:
        rmse = rmse ** 0.25
        # Plot red regions where Y == 1:
        plt.plot(max(rmse) * y, color='red', alpha=0.1, label='Anomaly')
        # Plot RMSE:
        plt.plot(rmse, label="RMSE", alpha=0.8, linewidth=0., color='blue', marker='o', markersize=1.)
        # Title:
        plt.title(f"{db.db_names[db_idx]} attack".replace('_', ' '))
        # Plot in 10_000 axline vertical:
        plt.axvline(TRAIN_LEN, color='black', alpha=0.5, linestyle='--', label='Stop AD training')
        # xlim between 0 and len(rmse):
        plt.xlim(0, len(rmse))
        plt.xlabel('Packet Number')
        plt.ylabel('$\log{(MSE)}$')
        plt.legend(loc='upper right')
        plt.show()
        break
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
