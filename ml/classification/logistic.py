import numpy as np
from ml.math.kernels import sigmoid
from ml.optimization.gradient_descent import SSEGradientDescent
from ml.optimization.regularization import L2Regularization

class LogisticRegressionModel:
    def __init__(self):
        self.params = []
        self.optimization = SSEGradientDescent()
        self.regularization = L2Regularization()

    def learn(self, x, y):
        m = x.shape[0]
        n = x.shape[1] + 1
        self.params = np.zeros(n)
        print(x.shape)
        X = np.concatenate((np.ones([m, 1]), x), axis=1)
        while not self.optimization.converged:
            p_update = self.optimization.step(X, self.predict(X)[1], y)
            #r_update = self.regularization.step(m, self.optimization.η)
            self.params[0] = self.params[0] - p_update[0]
            #self.params[1:] = self.params[1:] * r_update - p_update[1:]
            self.params[1:] -= p_update[1:]
            
    def predict(self, x):
        prob = sigmoid(x * self.params)
        return prob, round(prob)