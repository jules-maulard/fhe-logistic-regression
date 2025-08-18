import numpy as np

def bin_to_pm1(y_bin: np.ndarray) -> np.ndarray:
    return 2 * y_bin - 1

def add_bias(X: np.ndarray) -> np.ndarray:
    bias = np.ones((X.shape[0], 1))
    X_bias = np.hstack([bias, X])
    return X_bias

def pad_to_pow2_rows(mat: np.ndarray) -> np.ndarray:
    rows, cols = mat.shape
    next_rows = 2 ** int(np.ceil(np.log2(rows)))
    padded = np.zeros((next_rows, cols))
    padded[:rows, :] = mat
    return padded

def pad_to_pow2_cols(mat: np.ndarray) -> np.ndarray:
    rows, cols = mat.shape
    next_cols = 2 ** int(np.ceil(np.log2(cols)))
    padded = np.zeros((rows, next_cols))
    padded[:, :cols] = mat
    return padded

def init_wv(X, method='random'):
    n_features = X.shape[1]
    
    if method == 'zero':
        w = np.zeros(n_features)
    elif method == 'mean':
        w = np.mean(X, axis=0)
    elif method == 'random':
        w = np.random.randn(n_features) * 0.01

    return w, w.copy()
