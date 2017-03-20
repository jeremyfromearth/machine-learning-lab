import numpy as np

class SSEGradientDescent:
    '''
    Sum of Squared Errors Gradient Descent
    '''
    def __init__(self):
        self.complete = False
        self.η = 0.001
        self.mag = []
        self.tolerance = 0.1
        
    def step(self, x, h, y):
        e = h - y
        d = np.dot(e, x)
        m = sum(d) ** 2
        self.mag.append(m)
        self.complete = m < self.tolerance
        return self.η * (1 / x.shape[0]) * d
