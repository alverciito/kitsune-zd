# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import numpy as np
import pandas as pd


def preprocess_data(x_path, y_path: str | None, grant_benign: int, drop: list | tuple = ()):
    """
    Preprocess the data.
    :param x_path: The path to the data.
    :param y_path: The path to the labels. If None, the labels are assumed to be in the data.
    :param grant_benign: The number of benign samples to grant.
    :param drop: The columns to drop.
    :return: The preprocessed data and labels.
    """
    # Load the data
    x_df = pd.read_csv(x_path)
    if y_path is not None:
        y_df = pd.read_csv(y_path)
        if 'x' in y_df.columns:
            y_df = y_df['x']
        elif 'Label' in y_df.columns:
            y_df = y_df['Label']
        else:
            y_df = y_df.iloc[:, 0]
    else:
        y_df = x_df['Label']
        x_df = x_df.drop(columns=['Label'])
    x_df = x_df.drop(columns=list(drop))
    x = x_df.to_numpy().astype(np.float32)
    # Ensure that the benign samples are the first ones
    attype = dict()
    for data in y_df.unique():
        attype[data] = len(attype)
    y = y_df.map(attype).to_numpy().astype(np.float32)
    y = np.clip(y, 0, 1)
    # Take the first grant_benign samples
    benign_idx = np.where(y == 0)[0]
    if len(benign_idx) < grant_benign:
        raise ValueError(f"Number of benign samples is less than {grant_benign}.")
    benign_idx = benign_idx[:grant_benign]
    benign_x = x[benign_idx]
    benign_y = y[benign_idx]
    # Drop the benign samples from the dataset
    x = np.delete(x, benign_idx, axis=0)
    y = np.delete(y, benign_idx, axis=0)
    # Append the benign samples to the end of the dataset
    x = np.concatenate([benign_x, x], axis=0)
    y = np.concatenate([benign_y, y], axis=0)
    # Avoid nans in x:
    x = np.nan_to_num(x, nan=-1.0, posinf=+1e9, neginf=-1e9)
    # Assert that x has no invalid values:
    assert not np.any(np.isnan(x)), f"{x_path} has nan values."
    assert not np.any(np.isinf(x)), f"{x_path} has inf values."
    # Return the preprocessed data
    return x, y[:len(x)]

# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
