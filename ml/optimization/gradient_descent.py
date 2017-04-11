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
        
    def step(self, x, p, y):
        error = p - y
        delta = np.dot(error, x)
        mag = sum(delta) ** 2
        self.mag.append(mag)
        self.converged = mag < self.tolerance
        return (self.learning_rate / x.shape[0]) * delta