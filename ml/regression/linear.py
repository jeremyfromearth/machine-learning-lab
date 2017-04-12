import numpy as np
from ml.optimization.gradient_descent import SSEGradientDescent
from ml.optimization.regularization import L2Regularization

class LinearRegressionModel:
    def __init__(self):
        self.params = []
        self.cost_over_time = []
        self.max_iterations = 5000
        self.optimization = SSEGradientDescent()
        self.regularization = L2Regularization()
        
    def learn(self, x, y):
        iters = 0
        m = x.shape[0]
        n = x.shape[1] + 1
        self.cost_over_time = []
        self.params = np.zeros(n)
        X = np.concatenate((np.ones([m, 1]), x), axis=1)
        while not self.optimization.converged and iters < self.max_iterations:
            prediction = self.predict(X)
            self.cost_over_time.append(self.compute_cost(prediction, y))
            self.params -= self.optimization.step(X, prediction, y, self.regularization, self.params)
            iters += 1
            
    def predict(self, x):
        return np.dot(x, self.params)
    
    def compute_cost(self, prediction, y):
        return (1 / y.shape[0]) * np.sum((prediction - y) ** 2)