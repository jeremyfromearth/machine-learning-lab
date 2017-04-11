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
        X = np.concatenate((np.ones([m, 1]), x), axis=1)
        while not self.optimization.converged:
            p_update = self.optimization.step(X, self.predict(X)[1], y)
            r_update = self.regularization.step(m, self.optimization.learning_rate)
            self.params[0] = self.params[0] - p_update[0]
            self.params[1:] = self.params[1:] * r_update - p_update[1:]
            self.params[1:] -= p_update[1:]
            
    def predict(self, x):
        z = np.dot(self.params, x.T)
        prob = sigmoid(z)
        return prob, prob.round()
    
    def compute_cost(self, x, y):
        # not necessary for computing gradients, but interesting for research
        pass
    
    
class MultiClassLogisticRegression:
    def predict(self, x):
        #return predicted class
        pass
    pass