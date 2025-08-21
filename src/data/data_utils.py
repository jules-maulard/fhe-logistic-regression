import numpy as np
from sklearn.model_selection import train_test_split as split

import logging
logger = logging.getLogger(__name__)

def num_to_name(num: int):
    datasets = {
        1: "idash",
        2: "edinburgh", 
        3: "lbw",
        4: "nhanes3",
        5: "pcs",
        6: "uis"
    }    
    return datasets[num]

def num_to_data_path(num: int) -> str:
    return f"../data/{num_to_name(num)}.txt"

def load_data(data_path, shuffle=True):
    logger.info(f"Loading data from {data_path}")
    data = np.loadtxt(data_path, delimiter=",", skiprows=1)
    if shuffle:
        data = np.random.permutation(data)

    if data_path == num_to_data_path(1):
        y = data[:, 0]
        X = data[:, 1:]
    else:
        y = data[:, -1]
        X = data[:, :-1]

    logger.info(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features")
    return X, y

def train_test_split(X, y, test_size=0.2):   
    logger.info("Splitting data into train/test sets")
    X_train, X_test, y_train, y_test = split(
        X, 
        y, 
        test_size=test_size,
        random_state=42)
    logger.debug(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def stack_and_scale(scaler, X_train, X_test):
    logger.info("Stacking and scaling data")
    n_train = len(X_train)
    stacked = np.vstack([X_train, X_test])
    
    stacked_scaled = scaler.fit_transform(stacked)
    return stacked_scaled[:n_train], stacked_scaled[n_train:]