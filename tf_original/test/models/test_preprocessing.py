# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import tensorflow as tf
from src.old_models.preprocessing import WindowEvaluator, create_windowed_dataset, create_ar_windowed_dataset
from src.database import TestDatabase
FEATURE_SPACE = 115
TEST_LEN = 400_000_000_000
TRAIN_LEN = 200_000
SEQ_LEN = 100


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    # Create a new KitNET instance and database.
    db = TestDatabase(regularized=False)

    for db_idx, (x, y) in enumerate(db):
        # Format y:
        if len(x) != len(y):
            y = y[:len(x)]
        y = y[:TEST_LEN]
        x = x[:TEST_LEN]

        with tf.device('/GPU:1'):
            # Create the window evaluator:
            windowed = WindowEvaluator(FEATURE_SPACE, SEQ_LEN)
            dataset_window = create_windowed_dataset(x, SEQ_LEN, 1024)
            dataset_ar = create_ar_windowed_dataset(x, SEQ_LEN, 1024)
            len_window = 0
            len_ar = 0
            len_x = 0
            print('Len x: ', len(x))
            for x_batch, y_batch in dataset_window:
                if len_window == 0:
                    print('x shape: ', x_batch.shape)
                    print('y shape: ', y_batch.shape)
                len_window += len(x_batch)
            print('Len dataset window: ', len_window)
            for x_batch, y_batch in dataset_ar:
                if len_ar == 0:
                    print('x shape: ', x_batch.shape)
                    print('y shape: ', y_batch.shape)
                len_ar += len(x_batch)
            print('Len dataset ar: ', len_ar)
        print('EOP')
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
