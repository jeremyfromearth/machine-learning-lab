import numpy as np

class SSEGradientDescent:
    '''
    Sum of Squared Errors Gradient Descent
    '''
    def __init__(self):
        self.converged = False
        self.learning_rate = 0.0000001
        self.mag = []
        self.tolerance = 0.0000001
        
    def step(self, x, p, y, r, params):
        '''
            x - features
            p - predictions
            y - actual outputs
            r - regurlarization
            params - model params
        '''
        
        error = p - y
        delta = np.dot(error, x)
        regularization = r.derivative(x.shape[0], params[1:])
        gamma = (1 / x.shape[0]) * delta
        gamma[1:] = gamma[1:] + regularization
        new_params = self.learning_rate *  gamma
        
        mag = sum(delta) ** 2
        self.mag.append(mag)
        self.converged = mag < self.tolerance
        
        return new_params