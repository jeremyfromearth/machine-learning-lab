import numpy as np
from math import sqrt

class LinearRegressionModel:
    def __init__(self):
        self.weights = []
        self.eta = 0.001
        self.tolerance = 0.01
        self.iterations = 0

    def learn(self, data, features, target):
        # initialize member variables
        self.iterations = 0
        self.weights = np.zeros(1+len(features))
    
        # initialize gradient mag
        # this is a measure of the magnitude of the gradient of the aproximated function
        gradient_magnitude = float('inf')

        # gradient descent
        while gradient_magnitude > self.tolerance:
            # y-hat
            predictions = self.predict(data, features)
            # difference between predictions and output
            residuals = predictions - data[target]
            # accumulator for for gradient sum of squares
            gradient_sum_squares = 0.0
            # y-intercept
            self.weights[0] -= np.sum(residuals) * self.eta
            
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

            # measure the gradient
            gradient_magnitude = sqrt(gradient_sum_squares)
            # increment the iterations
            self.iterations += 1

    def predict(self, data, features):
        return np.dot(data[features], self.weights[1:]) + self.weights[0]

if __name__ == '__main__':
    import pandas as pd
    features = ['x']
    target = 'y'
    data = pd.DataFrame(
        data=list(zip([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0
        ], 
        [
            4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0
        ])), columns=['x', 'y'])

    model = LinearRegressionModel()
    model.learn(data, features, target)
    test = pd.DataFrame(data=list(zip([11.0], [24.0])), columns=['x', 'y'])
    predictions = model.predict(test, features)
    print('Predictions: {}, Iterations; {}, Weights: {}'.format(predictions, model.iterations, model.weights))
    


    
