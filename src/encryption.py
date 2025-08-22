import numpy as np
from openfhe import *

import logging
logger = logging.getLogger(__name__)

def generate_rotation_keys(cc, sk, n, f):
    log_f = int(np.log2(f + 1))
    log_nf = int(np.log2(n * (f + 1)))

    pos_shifts = [2**j for j in range(log_nf)]
    neg_shifts = [-2**j for j in range(log_f)]

    all_shifts = sorted(set(pos_shifts + neg_shifts))

    cc.EvalAtIndexKeyGen(sk, all_shifts)

def init_ckks(n, f, mult_depth=6, scaling_mod_size=30):
    logger.debug("Initializing CKKS context")
    logger.debug(f"Input parameters: n={n}, f={f}, mult_depth={mult_depth}")
    
    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(mult_depth)
    params.SetScalingModSize(scaling_mod_size)
    params.SetBatchSize(n * (f + 1))
    logger.debug(f"CKKS parameters: {params}")

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)

    key_pair = cc.KeyGen()
    cc.EvalMultKeyGen(key_pair.secretKey)
    generate_rotation_keys(cc, key_pair.secretKey, n, f)

    logger.info("CKKS context initialized.")
    return cc, key_pair, params


def encode_encrypt_vector(vec: np.ndarray, cc, public_key) -> object:
    plaintext = cc.MakeCKKSPackedPlaintext(vec)
    ciphertext = cc.Encrypt(public_key, plaintext)
    return ciphertext

def encode_encrypt_features(Z: np.ndarray, cc, public_key) -> object:
    logger.debug(f"Input shape: {Z.shape}")

    Z_vec = Z.flatten()
    ciphertext = encode_encrypt_vector(Z_vec, cc, public_key)

    logger.debug(f"Encrypted vector length: {len(Z_vec)}")
    logger.info("Features encrypted.")
    return ciphertext, len(Z_vec)

def decrypt_decode_matrix(
    ciphertext,
    cc,
    secret_key,
    vec_length: int,
    original_shape: tuple[int, int]
) -> np.ndarray:
    decrypted = cc.Decrypt(secret_key, ciphertext)
    decrypted.SetLength(vec_length)
    arr = np.array(decrypted.GetRealPackedValue())

    logger.debug(f"Output shape: {original_shape}")
    logger.info("Matrix decrypted.")
    return arr.reshape(original_shape)

def encode_encrypt_beta(beta: np.ndarray, cc, public_key, n) -> object:
    logger.debug(f"Beta shape: {beta.shape}, n: {n}")
    packed_beta = np.tile(beta, n)
    ciphertext = encode_encrypt_vector(beta, cc, public_key)
    logger.info("Beta encrypted.")
    return ciphertext