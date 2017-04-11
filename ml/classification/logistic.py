import numpy as np
from ml.math.kernels import sigmoid
from ml.optimization.gradient_descent import SSEGradientDescent
from ml.optimization.regularization import L2Regularization

class LogisticRegressionModel:
    def __init__(self):
        self.params = []
        self.optimization = SSEGradientDescent()
        self.regularization = L2Regularization()
        self.cost_over_time = []
        self.params_over_time = []

    def learn(self, x, y):
        m = x.shape[0]
        n = x.shape[1] + 1
        self.cost_over_time = []
        self.params = np.zeros(n)
        self.params_over_time = []
        X = np.concatenate((np.ones([m, 1]), x), axis=1)
        while not self.optimization.converged:
            prediction = self.predict(X)
            self.cost_over_time.append(self.compute_cost(prediction[0], y))
            p_update = self.optimization.step(X, prediction[0], y)
            r_update = self.regularization.step(m, self.optimization.learning_rate)
            self.params[0] = self.params[0] - p_update[0]
            self.params[1:] = self.params[1:] * r_update - p_update[1:]
            self.params_over_time.append(list(self.params))
            
    def predict(self, x):
        z = np.dot(x, self.params)
        prob = sigmoid(z)
        return prob, prob.round()
    
    def compute_cost(self, prediction, y):
        element_wise_cost = y * np.log(prediction) + (1 - y) * np.log(1-prediction)
        cost = -(1.0 / y.shape[0]) * np.sum(element_wise_cost)
        return cost