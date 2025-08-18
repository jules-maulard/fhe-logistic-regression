import numpy as np

import logging
logger = logging.getLogger(__name__)

LSQ_COEFFS = {
    3: [0.5, -0.15012, 0.001593], 
    5: [0.5, -0.19131, 0.0045963, -0.0000412332],
    7: [0.5, -0.216884, 0.00819276, -0.000165861, 0.00000119581]
}

def sigmoid(x):
    val = 1 / (1 + np.exp(-x))
    logger.debug(f"sigmoid output: [{val.min()}, {val.max()}]")
    return val

def minus_sigmoid(x):
    val = 1 / (1 + np.exp(x))
    logger.debug(f"minus sigmoid output: [{val.min()}, {val.max()}]")
    return val



def lsq3(x, clip=False):
    logger.debug(f"lsq input: [{x.min()}, {x.max()}]")
    a0, a1, a3 = LSQ_COEFFS[3]
    val = a0 + a1 * x + a3 * x**3
    logger.debug(f"lsq approx: [{val.min()}, {val.max()}]")
    return np.clip(val, 0, 1) if clip else val

def lsq5(x, clip=False):
    logger.debug(f"lsq input: [{x.min()}, {x.max()}]")
    a0, a1, a3, a5 = LSQ_COEFFS[5]
    val = a0 + a1 * x + a3 * x**3 + a5 * x**5
    logger.debug(f"lsq approx: [{val.min()}, {val.max()}]")
    return np.clip(val, 0, 1) if clip else val

def lsq7(x, clip=False):
    logger.debug(f"lsq input: [{x.min()}, {x.max()}]")
    a0, a1, a3, a5, a7 = LSQ_COEFFS[7]
    val = a0 + a1 * x + a3 * x**3 + a5 * x**5 + a7 * x**7
    logger.debug(f"lsq approx: [{val.min()}, {val.max()}]")
    return np.clip(val, 0, 1) if clip else val



def lsq3_ckks(ct_ip, cc):
    a0, a1, a3 = LSQ_COEFFS[3]
    ct_x = ct_ip
    ct_x2 = cc.EvalMult(ct_x, ct_x)
    ct_x2 = cc.ModReduce(ct_x2)
    ct_x3 = cc.EvalMult(ct_x, ct_x2)  
    ct_x3 = cc.ModReduce(ct_x3) # x^3

    ct_sig = cc.EvalMult(ct_x, a1)
    ct_sig = cc.ModReduce(ct_sig) # a1 * x

    ct_sig = cc.EvalAdd(ct_sig, a0) # + a0

    ct_x3a3 = cc.EvalMult(ct_x3, a3)
    ct_x3a3 = cc.ModReduce(ct_x3a3)
    ct_sig = cc.EvalAdd(ct_sig, ct_x3a3) # + a3 * x^3

    return ct_sig

def lsq5_ckks(ct_ip, cc):
    a0, a1, a3, a5 = LSQ_COEFFS[5]
    ct_x = ct_ip
    ct_x2 = cc.EvalMult(ct_x, ct_x)
    ct_x2 = cc.ModReduce(ct_x2)
    ct_x3 = cc.EvalMult(ct_x, ct_x2)  # x^3
    ct_x3 = cc.ModReduce(ct_x3)
    ct_x5 = cc.EvalMult(ct_x3, ct_x2)  # x^5
    ct_x5 = cc.ModReduce(ct_x5)
    
    ct_sig = cc.EvalMult(ct_x, a1)
    ct_sig = cc.ModReduce(ct_sig)
    ct_sig = cc.EvalAdd(ct_sig, a0)
    
    ct_x3a3 = cc.EvalMult(ct_x3, a3)
    ct_x3a3 = cc.ModReduce(ct_x3a3)
    ct_sig = cc.EvalAdd(ct_sig, ct_x3a3)
    
    ct_x5a5 = cc.EvalMult(ct_x5, a5)
    ct_x5a5 = cc.ModReduce(ct_x5a5)
    ct_sig = cc.EvalAdd(ct_sig, ct_x5a5)
    
    return ct_sig

def lsq7_ckks(ct_ip, cc):
    a0, a1, a3, a5, a7 = LSQ_COEFFS[7]
    ct_x = ct_ip
    ct_x2 = cc.EvalMult(ct_x, ct_x)
    ct_x2 = cc.ModReduce(ct_x2)
    ct_x3 = cc.EvalMult(ct_x, ct_x2)  # x^3
    ct_x3 = cc.ModReduce(ct_x3)
    ct_x5 = cc.EvalMult(ct_x3, ct_x2)  # x^5
    ct_x5 = cc.ModReduce(ct_x5)
    ct_x7 = cc.EvalMult(ct_x5, ct_x2)  # x^7
    ct_x7 = cc.ModReduce(ct_x7)
    
    ct_sig = cc.EvalMult(ct_x, a1)
    ct_sig = cc.ModReduce(ct_sig)
    ct_sig = cc.EvalAdd(ct_sig, a0)
    
    ct_x3a3 = cc.EvalMult(ct_x3, a3)
    ct_x3a3 = cc.ModReduce(ct_x3a3)
    ct_sig = cc.EvalAdd(ct_sig, ct_x3a3)
    
    ct_x5a5 = cc.EvalMult(ct_x5, a5)
    ct_x5a5 = cc.ModReduce(ct_x5a5)
    ct_sig = cc.EvalAdd(ct_sig, ct_x5a5)
    
    ct_x7a7 = cc.EvalMult(ct_x7, a7)
    ct_x7a7 = cc.ModReduce(ct_x7a7)
    ct_sig = cc.EvalAdd(ct_sig, ct_x7a7)
    
    return ct_sig



_SIGMOID_APPROX = {
    ("sigmoid", 0, False): sigmoid,
    ("minus_sigmoid", 0, False): minus_sigmoid,

    ("least_squares", 3, False): lsq3,
    ("least_squares", 5, False): lsq5,
    ("least_squares", 7, False): lsq7,

    ("least_squares", 3, True): lsq3_ckks,
    ("least_squares", 5, True): lsq5_ckks,
    ("least_squares", 7, True): lsq7_ckks,

    # ("chebyshev", 3): chebyshev3,
    # ("chebyshev", 5): chebyshev5,
}

def get_sigmoid_approx(method="sigmoid", degree=0, encrypted=False):
    logger.debug(f"Getting sigmoid approximation: method={method}, degree={degree}, encrypted={encrypted}")
    try:
        return _SIGMOID_APPROX[(method, degree, encrypted)]
    except KeyError:
        raise ValueError(f"No sigmoid approximation for method={method}, degree={degree}, encrypted={encrypted}")