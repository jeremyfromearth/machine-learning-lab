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
        self.max_iterations = 5000

    def learn(self, x, y):
        iters = 0
        m = x.shape[0]
        n = x.shape[1] + 1
        self.cost_over_time = []
        self.params = np.zeros(n)
        self.params_over_time = []
        X = np.concatenate((np.ones([m, 1]), x), axis=1)
        while not self.optimization.converged and iters < self.max_iterations:
            prediction = self.predict(X)
            self.cost_over_time.append(self.compute_cost(prediction[0], y))
            p_update = self.optimization.step(X, prediction[0], y, self.regularization, self.params)
            self.params[0] = self.params[0] - p_update[0]
            self.params[1:] = self.params[1:] - p_update[1:]
            self.params_over_time.append(list(self.params))
            iters += 1
            
    def predict(self, x):
        z = np.dot(x, self.params)
        prob = sigmoid(z)
        return prob, prob.round()
    
    def compute_cost(self, prediction, y):
        element_wise_cost = y * np.log(prediction) + (1 - y) * np.log(1-prediction)
        regularization_term = self.regularization.cost(y.shape[0], self.params[1:])
        cost = -(1.0 / y.shape[0]) * np.sum(element_wise_cost) + regularization_term
        return cost