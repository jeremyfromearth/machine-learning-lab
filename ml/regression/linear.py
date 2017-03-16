import numpy as np
from math import sqrt
    
class SSEGradientDescent:
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
    
class PassThruRegularization:
    def step(self, m, η):
        return 1.0
    
class L2Regularization:
    def __init__(self):
        self.λ = 0.05
    
    def step(self, m, η):
        return 1 - (η * (self.λ / m))
        
class LinearRegressionModel2:
    def __init__(self):
        self.params = []
        self.iterations = 0
        self.cost_histogram = []
        self.optimization = SSEGradientDescent()
        self.regularization = PassThruRegularization()
        
    def optimize(self, x, y):
        m = x.shape[0]
        n = x.shape[1] + 1
        self.params = np.zeros(n)
        X = np.concatenate((np.ones([m, 1]), x), axis=1)
        while not self.optimization.complete:
            p_update = self.optimization.step(X, self.predict(X), y)
            r_update = self.regularization.step(m, self.optimization.η)
            self.params[0] = self.params[0] - p_update[0]
            self.params[1:] = self.params[1:] * r_update - p_update[1:]
    
    def predict(self, x):
        return np.dot(x, self.params)

class LinearRegressionModel:
    def __init__(self):
        self.weights = []
        self.eta = 0.001
        self.tolerance = 0.1
        self.iterations = 0
        self.rss = []
        self.gradient_magnitudes = []

    def learn(self, data, features, target):
        '''
        Trains the model 

        Parameters
        ---------
        data : pd.DataFrame
            N-Dimensional array of data
        features : list
            A list of strings indicating which columns represent features
        target : string
            The name of the column to predict
        '''
        # initialize member variables
        self.iterations = 0
        self.residuals = []
        self.weights = np.zeros(1+len(features))
    
        # initialize gradient mag
        # initialize a list to store magnitudes over iterations
        # this is a measure of the magnitude of the gradient of the aproximated function
        self.gradient_magnitudes = []
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
            self.weights[0] -= self.eta * np.sum(residuals)
            
            # feature by feature update the weights
            for i in range(len(features)):
                # feature at i, convert to nparray - it's faster
                xi = data[features[i]].values
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
        '''
        Returns a list of predictions for the supplied data

        Parameters
        ----------
        data : pd.DataFrame
            A dataframe containg data to make predictions on
        features : list
            A list of columns names to base predictions on
        '''
        return np.dot(data[features], self.weights[1:]) + self.weights[0]

    def get_squared_error_loss(self, data, features, target):
        '''
        Returns the squared error loss for the supplied data

        Parameters
        ----------
        data : pd.DataFrame
            Data to assess squared error on
        features : list
            A list of column names to base assessment on
        target : string
            Name of the column to predict
        '''
        predictions = self.predict(data, features)
        errors = data[target] - predictions
        return (1.0 / len(data)) * np.sum(errors **2)

