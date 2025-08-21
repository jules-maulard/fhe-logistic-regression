import numpy as np

import logging
logger = logging.getLogger(__name__)


class LogisticRegressionIDASH:
    def __init__(
        self,
        sigmoid,
        # NAG
        use_NAG: bool = False,
        nag_momentum_update = None
    ):
        self.sigmoid = sigmoid
        # NAG
        self.use_NAG = use_NAG
        self.nag_momentum_update = nag_momentum_update

        self.pt_beta = None
        self.beta_history = []

    def fit(
        self, 
        pt_Z: np.ndarray, Z_dims, 
        pt_beta: np.ndarray, 
        # Optimisation
        n_epoch: int = 10,
        gamma_up: float = 10,
        gamma_down: float = 1,
        # NAG
        pt_v: np.ndarray = None, 
    ):
        n, f = Z_dims
        
        for epoch in range(1, n_epoch + 1):
            logger.info(f">>> Epoch {epoch+1} <<<")

            alpha_t = self._update_lr(epoch, n, gamma_up, gamma_down)
            gamma = self.nag_momentum_update.update() if self.use_NAG else 0
            logger.debug(f"Learning rate: {alpha_t * n:.4f} - Smoothing parameter: {gamma}")
            pt_eval_beta = pt_v if self.use_NAG else pt_beta
            
            pt_ip = np.dot(pt_Z, pt_eval_beta)
            pt_activated = self.sigmoid(pt_ip)
            pt_gradient = alpha_t * (np.dot(pt_activated, pt_Z))

            if self.use_NAG:
                tmpw = pt_v + pt_gradient
                pt_v = (1 - gamma) * tmpw + gamma * pt_beta
                pt_beta = tmpw
            else:
                pt_beta += pt_gradient
            self.beta_history.append(pt_beta.copy())

        self.pt_beta = pt_beta 


    def _update_lr(self, iteration, sampleDimTrain, gammaUp, gammaDown):
        if gammaDown > 0:
            lr = gammaUp / gammaDown / sampleDimTrain
        else:
            lr = gammaUp / (iteration - gammaDown) / sampleDimTrain
        return lr

    def predict_proba(self, pt_Z, beta, sigmoid=None):
        sigmoid = self.sigmoid if sigmoid is None else sigmoid

        pt_ip = np.dot(pt_Z, beta)
        pt_y_proba = 1 - sigmoid(pt_ip)
        return pt_y_proba
