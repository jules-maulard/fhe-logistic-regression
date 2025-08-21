import numpy as np
import math

import logging
logger = logging.getLogger(__name__)

def bin_to_pm1(y_bin: np.ndarray) -> np.ndarray:
    return 2 * y_bin - 1

def add_bias(X: np.ndarray) -> np.ndarray:
    logger.info(f"Adding bias term")
    original_shape = X.shape
    bias = np.ones((X.shape[0], 1))
    X_bias = np.hstack([bias, X])
    logger.debug(f"Added bias term: {original_shape} -> {X_bias.shape}")
    return X_bias

def pad_to_pow2_rows(mat: np.ndarray) -> np.ndarray:
    rows, cols = mat.shape
    next_rows = 2 ** int(np.ceil(np.log2(rows)))
    logger.info(f"Padding rows: {rows} -> {next_rows} (next power of 2)")
    padded = np.zeros((next_rows, cols))
    padded[:rows, :] = mat
    return padded

def pad_to_pow2_cols(mat: np.ndarray) -> np.ndarray:
    rows, cols = mat.shape
    next_cols = 2 ** int(np.ceil(np.log2(cols)))
    logger.info(f"Padding columns: {cols} -> {next_cols} (next power of 2)")
    padded = np.zeros((rows, next_cols))
    padded[:, :cols] = mat
    return padded

def init_wv(X, method='random'):
    n_features = X.shape[1]
    logger.info(f"Initializing weights for {n_features} features as: '{method}'")

    valid_methods = ['zero', 'mean', 'mean-idash', 'random']
    if method not in valid_methods:
        logger.warning(f"Unknown method '{method}'. Defaulting to 'random'")
        method = 'random'
    
    if method == 'zero':
        w = np.zeros(n_features)
    elif method == 'mean':
        w = np.mean(X, axis=0)
    elif method == 'mean-idash':
        w = np.mean(X, axis=0)
        sdim_bits = math.ceil(math.log2(X.shape[0]))
        sdim_pow = 1 << sdim_bits
        w = w / sdim_pow
    elif method == 'random':
        w = np.random.randn(n_features) * 0.01

    logger.debug(f"Weight stats - Min: {w.min():.4f}, Max: {w.max():.4f}, Mean: {w.mean():.4f}")
    return w, w.copy()
