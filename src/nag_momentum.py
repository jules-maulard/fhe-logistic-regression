import math

class NAGUpdater:
    def __init__(
        self, 
        version='basic', 
        alpha0=1.0, 
        t=0, 
        gamma_cst=0.9
    ):
        self.version = version
        self.params = {
            'alpha0': alpha0,
            'alpha1': alpha0,
            't': t,
            'gamma_cst': gamma_cst
        }
        
        self.updaters = {
            'basic': self._basic_update,
            'idash': self._idash_update,
            'idash_correct': self._idash_correct_update,
            'constant': self._constant_update
        }
    
    def update(self):
        return self.updaters[self.version]()
    


    def _basic_update(self):
        self.params['t'] += 1
        t = self.params['t']
        return (t - 1) / (t + 2)
    
    def _idash_update(self):
        def update_alpha(alpha):
            return (1 + math.sqrt(1 + 4 * alpha * alpha)) / 2
        
        alpha0 = self.params['alpha0']
        alpha1 = self.params['alpha1']

        alpha0 = alpha1
        alpha1 = update_alpha(alpha0)
        gamma = (1 - alpha0) / alpha1
        
        self.params.update({'alpha0': alpha0, 'alpha1': alpha1})
        return gamma
    
    def _idash_correct_update(self):
        def update_alpha(alpha):
            return (1 + math.sqrt(1 + 4 * alpha * alpha)) / 2
        
        alpha0 = self.params['alpha0']
        alpha1 = self.params['alpha1']

        alpha0 = alpha1
        alpha1 = update_alpha(alpha0)
        gamma = 1 + ((1 - alpha0) / alpha1)
        
        self.params.update({'alpha0': alpha0, 'alpha1': alpha1})
        return gamma
    
    def _constant_update(self):
        return self.params['gamma_cst']
    