import numpy as np

import logging
logger = logging.getLogger(__name__)


class LogisticRegression:
    def __init__(
        self,
        sigmoid,
        # FHE
        cc,
        # NAG
        use_NAG: bool = False,
        nag_momentum_update = None
    ):
        self.sigmoid = sigmoid
        # FHE
        self.cc = cc
        # NAG
        self.use_NAG = use_NAG
        self.nag_momentum_update = nag_momentum_update

        self.ct_beta = None
        self.ct_beta_history = []

    def fit(
        self, 
        ct_Z: np.ndarray, Z_dims, 
        ct_beta: np.ndarray, 
        # Optimisation
        n_epoch: int = 10,
        gamma_up: float = 10,
        gamma_down: float = 1,
        # NAG
        ct_v: np.ndarray = None, 
    ):
        n, f = Z_dims
        
        for epoch in range(1, n_epoch+1):
            logger.info(f">>> Epoch {epoch} <<<")

            alpha_t = self._update_lr(epoch, n, gamma_up, gamma_down)
            gamma = self.nag_momentum_update.update() if self.use_NAG else 0
            logger.debug(f"Learning rate: {alpha_t * n:.4f} - Smoothing parameter: {gamma}")
            ct_eval_beta = ct_v if self.use_NAG else ct_beta

            ct_ip = inner_product(ct_Z, ct_eval_beta, self.cc, n, f)
            ct_activated = self.sigmoid(ct_ip, self.cc)
            ct_gradient = aggregate_gradient(ct_activated, ct_Z, self.cc, n, f)

            ct_beta_new = update_beta(ct_gradient, ct_eval_beta, alpha_t, self.cc, n, f)
            if self.use_NAG:
                ct_v = update_v(ct_beta, ct_beta_new, gamma, self.cc, n, f)
            self.ct_beta_history.append(ct_beta)

        self.ct_beta = ct_beta 


    def _update_lr(self, iteration, sampleDimTrain, gammaUp, gammaDown):
        if gammaDown > 0:
            lr = (gammaUp / gammaDown) / sampleDimTrain
        else:
            lr = (gammaUp / (iteration - gammaDown)) / sampleDimTrain
        return lr

    def predict_proba(self, ct_Z, beta, sigmoid=None):
        sigmoid = self.sigmoid if sigmoid is None else sigmoid

        ct_ip = np.dot(ct_Z, beta)
        ct_y_proba = 1 - sigmoid(ct_ip)
        return ct_y_proba





def inner_product(ct_Z, ct_beta, cc, n, f) -> object:
    logger.debug("Computing inner product")
    ct_ip = cc.EvalMult(ct_Z, ct_beta)
    ct_ip = cc.ModReduce(ct_ip)

    rotations = int(np.log2(f + 1))
    for j in range(rotations):
        rotated = cc.EvalAtIndex(ct_ip, 2**j)
        ct_ip = cc.EvalAdd(ct_ip, rotated)

    pt_mask = make_mask(n, f, cc)
    ct_ip = cc.EvalMult(ct_ip, pt_mask)
    ct_ip = cc.ModReduce(ct_ip)

    for j in range(rotations):
        rotated = cc.EvalAtIndex(ct_ip, -(2**j))
        ct_ip = cc.EvalAdd(ct_ip, rotated)

    logger.info("Inner product computed.")
    return ct_ip

def make_mask(n, f, cc):
    line_length = f + 1
    total_slots = n * line_length
    mask = [0] * total_slots
    for i in range(n):
        mask[i * line_length] = 1
    plaintext_mask = cc.MakeCKKSPackedPlaintext(mask, noiseScaleDeg=20)
    logger.debug(f"Mask created: length {len(mask)}")
    return plaintext_mask


def aggregate_gradient(ct_activated, ct_Z, cc, n, f):
    logger.debug("Computing aggregate gradient")

    grad = cc.EvalMult(ct_activated, ct_Z)
    grad = cc.ModReduce(grad)

    log_f = int(np.log2(f + 1))
    log_nf = log_f + int(np.log2(n))
    for j in range(log_f, log_nf):
        rotated = cc.EvalAtIndex(grad, 2**j)
        grad = cc.EvalAdd(grad, rotated)
        
    logger.info("Aggregate gradient computed.")
    return grad


def update_beta(ct_grad, ct_beta, alpha_t, cc, n, f):
    logger.debug("Update beta")

    # pc = 20
    scaled_alpha = alpha_t # * (2 ** pc)
    pt_scaled_alpha = cc.MakeCKKSPackedPlaintext([scaled_alpha] * (n*(f+1))) # , scaleDeg=30)
    grad = cc.EvalMult(ct_grad, pt_scaled_alpha)
    grad = cc.ModReduce(grad)

    ct_beta_updated = cc.EvalAdd(ct_beta, grad)

    logger.info("Beta updated.")
    return ct_beta_updated

def update_v(ct_beta, ct_beta_new, gamma, cc, n, f):
    logger.debug("Update v")

    delta1 = cc.MakeCKKSPackedPlaintext([gamma] * (n*(f+1)))
    delta2 = cc.MakeCKKSPackedPlaintext([1 -gamma] * (n*(f+1)))
    ct_beta1 = cc.EvalMult(delta1, ct_beta)
    ct_beta1 = cc.ModReduce(ct_beta1)
    ct_beta2 = cc.EvalMult(delta2, ct_beta_new)
    ct_beta2 = cc.ModReduce(ct_beta2)
    
    ct_v_updated = cc.EvalAdd(ct_beta2,ct_beta1)
    ct_v_updated = cc.ModReduce(ct_v_updated)

    logger.info("v updated.")
    return ct_v_updated