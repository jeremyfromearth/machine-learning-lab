import numpy as np

class PassThruRegularization:
    def derivative(self, m, params):
        return 0.0
    
class L2Regularization:
    def __init__(self):
        self.param = 10.0
    
    def derivative(self, m, params):
        '''
            Calculates the derivative of the regularization function
            Useful in conjunction with gradient descent
            m - number of training examples
            params - model parameters
        '''
        return (self.param/m) * params
    
    def cost(self, m, params):
        '''
            Calculates the regularization values to be applied to cost function
            m - number of training examples
            params - model parameters
        '''
        return self.param / (2 * m) * np.sum(params ** 2)
