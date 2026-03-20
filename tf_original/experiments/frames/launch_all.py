# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import tqdm
import numpy as np
from src.models import KitNET, NETWORK_TYPES
import logging
import os
import tensorflow as tf

CONFIG = {
    # Train relevant:
    'train_period': 150,  # kPacket
    'clustering_period': 50, # kPacket
    # Model relevant:
    'sequence_length': 800, # Packet
    'hidden_ratio': 0.22,
    'autoencoder_size': 4,
    'clustering': 'dbscan',
    'output_ae_type': 'stat',
    # Performance relevant:
    'execution_window': 400, # kPacket
}


def model_experiment_launcher(x: np.ndarray, y: np.ndarray, path: str):
    """
    This function launches the experiment with the given data.
    :param x: The data to process.
    :param y: The labels of the data.
    :param path: The path to save the results.
    :return: None
    """
    # Check x, y:
    if len(x) != len(y):
        y = y[:len(x)]
        logging.warning(f'The labels have been truncated to match the data length: {len(x)} != {len(y)}')
    # Check path:
    if not os.path.exists(path):
        raise ValueError(f'The path: {path} does not exist.')

    logging.info(f'Lanching experiment with data of shape: {x.shape} and path: {path}')
    with tf.device('/GPU:1'):
        for is_ar in [True, False]:
            for model_type in NETWORK_TYPES:
                # Check if the model has already been processed:
                if not os.path.exists(f'{path}/{model_type}_ar_{is_ar}.npy'):
                    logging.info(f'Processing model: {model_type}')
                    kn = KitNET(x.shape[-1],
                                feature_map=None,
                                ad_grace_period=1000 * CONFIG['train_period'],
                                execution_window=1000 * CONFIG['execution_window'],
                                fm_grace_period=1000 * CONFIG['clustering_period'],
                                sequence_length=CONFIG['sequence_length'],
                                hidden_ratio=CONFIG['hidden_ratio'],
                                autoencoder_size=CONFIG['autoencoder_size'],
                                clustering=CONFIG['clustering'],
                                ar=is_ar,
                                ae_type=model_type,
                                output_ae_type=CONFIG['output_ae_type'])


                    # Process packet with KitNet:
                    with tqdm.tqdm(total=(len(x)), desc='Processing data') as pbar:
                        for _, packet in enumerate(x):
                            kn.process(packet, is_last=False if _ < len(x) - 1 else True)
                            pbar.update(1)
                    try:
                        kn.show_stats(y, save=f'{path}/{model_type}_ar_{is_ar}')
                    except Exception as e:
                        logging.error(f'Error processing model {model_type} / dismiss.')
                        logging.error(e)
                else:
                    logging.info(f'Model {model_type} already processed.')


def data_loader(path: str, skip: int = None) -> (pd.DataFrame, dict):
    """
    This function loads the data from the given path.
    """
    if skip is not None:
        df = pd.read_csv(path, skiprows=lambda x: x == skip)
    else:
        df = clean_csv_headers(path)

    if len(df.columns) == 1:
        if skip is not None:
            df = pd.read_csv(path, skiprows=lambda x: x == skip)
        else:
            df = clean_csv_headers(path, delimiter=';')

    # Drop timestamps and nan:
    df = df.fillna(0)
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])

    label_encoders = dict()
    are_categories = list()
    label_name = None

    for column in df.columns:
        # Encode:
        label_enc = LabelEncoder()
        if df[column].dtype == 'object':
            df[column] = label_enc.fit_transform(df[column])
            label_encoders[column] = label_enc
            if column not in ['Label', 'label', 'type', 'Type', 'attack_type', 'Attack_type', ' Label']:
                are_categories.append(True)
            else:
                label_name = column
        else:
            are_categories.append(False)
        # Replace inf:
        max_non_inf = df[column][np.isfinite(df[column])].max()
        df[column] = df[column].replace([np.inf, -np.inf], max_non_inf)

    # Raise exception if label_name is None:
    if label_name is None:
        raise ValueError(f'The label name could not be found in the dataset: {df.columns}')

    # Get x, y:
    x = df.drop(columns=[label_name]).values if label_name in df.columns else df.drop(columns=[label_name]).values
    y = df[label_name].values if label_name in df.columns else df[label_name].values

    return (x, y), (label_encoders, are_categories)

def regularize_data(x: np.ndarray, y: np.ndarray, are_categorical: list[bool]) -> tuple:
    """
    This function regularizes the data.
    :param x: The data to regularize.
    :param y: The labels of the data.
    :param are_categorical: The list of booleans that indicates which columns are categorical.
    :return:
    """
    # Regularize y:
    benign_labels = np.where(y == y[0])[0]
    malicious_labels = np.where(y != y[0])[0]
    y[benign_labels] = 0
    y[malicious_labels] = 1
    # Regularize x:
    are_categorical = np.array(are_categorical)
    categorical_x = x[:, are_categorical]
    numerical_x = x[:, ~are_categorical]
    # Encode categorical data:
    categorical_x = categorical_x.astype(np.uint8)
    categorical_x = np.unpackbits(categorical_x, axis=1)
    # Regularize numerical data:
    numerical_x = (numerical_x - numerical_x.mean(axis=0)) / (numerical_x.std(axis=0) + 1e-6)
    # Mix the data:
    x = np.concatenate([numerical_x, categorical_x], axis=1)
    return x, y


def grant_first_label(x: np.ndarray, y: np.ndarray, size: int) -> (np.ndarray, np.ndarray):
    """
    This function grants the first 'size' elements of the dataset to contain the first label.
    :param x: The data.
    :param y: The labels.
    :param size: The size of the first label.
    """
    first_label = y[0]

    # Get the first 'size' that match the first label:
    args_y = np.argwhere(y == first_label).flatten()
    if len(args_y) < size:
        raise ValueError(f'The first label {first_label} does not have enough samples: {len(args_y)} < {size}')
    else:
        args_y = args_y[:size]

    first_x = x[args_y]
    first_y = y[args_y]
    # Append the rest of the data:
    new_x = np.concatenate([first_x, x])
    new_y = np.concatenate([first_y, y])
    return new_x, new_y

def clean_csv_headers(file_path, delimiter=','):
    """
    This function cleans the headers of the CSV file.
    :param file_path: The path to the CSV file.
    :param delimiter: The delimiter of the CSV file.
    :return: The cleaned DataFrame.
    """
    df = pd.read_csv(file_path, delimiter=delimiter)
    # Header column:
    header = df.columns
    # Drop SimilarHTTP header:
    if 'SimillarHTTP' in header:
        df = df.drop(columns=['SimillarHTTP'])
    # Return the DataFrame:
    return df
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
