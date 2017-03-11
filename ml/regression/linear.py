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
        '''
        Trains the model 

        Parameters
        ---------
        data : pd.DataFrame
            N-1Dimensional array of data
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

