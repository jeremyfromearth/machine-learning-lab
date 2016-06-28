import numpy as np
from math import sqrt

class LinearRegressionModel:
    def __init__(self):
        self.weights = []
        self.eta = 0.001
        self.tolerance = 0.01
        self.iterations = 0
        self.rss = []
        self.gradient_magnitudes = []

    def learn(self, data, features, target):
        # initialize member variables
        self.iterations = 0
        self.residuals = []
        self.weights = np.zeros(1+len(features))
    
        # initialize gradient mag
        # this is a measure of the magnitude of the gradient of the aproximated function
        gradient_magnitude = float('inf')
        self.gradient_magnitudes = []

        # gradient descent
        while gradient_magnitude > self.tolerance:
            # y-hat
            predictions = self.predict(data, features)
            # difference between predictions and output
            residuals = predictions - data[target]
            # accumulator for for gradient sum of squares
            gradient_sum_squares = 0.0
            # y-intercept
            self.weights[0] -= self.eta * np.sum(residuals)
            
            # feature by feature update the weights
            for i in range(len(features)):
                # feature at i
                xi = data[features[i]]
                # the partial derivate with respect to xi
                partial = 2 * np.dot(residuals, xi) 
                # add the partial squared
                gradient_sum_squares += partial ** 2
                # set the new w-hat
                self.weights[i+1] -= self.eta * partial

            # store the residuals per each iteration
            self.rss.append(np.sum(residuals ** 2))
            # measure the gradient
            gradient_magnitude = sqrt(gradient_sum_squares)
            self.gradient_magnitudes.append(gradient_magnitude)
            # increment the iterations
            self.iterations += 1

    def predict(self, data, features):
        return np.dot(data[features], self.weights[1:]) + self.weights[0]

    def get_squared_error_loss(self, data, features, target):
        predictions = self.predict(data, features)
        errors = data[target] - predictions
        return (1.0 / len(data)) * np.sum(errors **2)

