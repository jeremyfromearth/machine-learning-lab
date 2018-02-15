import numpy as np
from ml.kernels import sigmoid, relu

class NeuralNetwork:
    def __init__(self, layers, rseed=0, activation=sigmoid):
      np.random.seed(rseed)
      self.e = 0.1
      self.a = {}
      self.z = {}
      self.b = {}
      self.w = {}

      # append an extra layer here
      # will need to change this in the event that we are doing multi-class classification
      self.nl = len(layers)
      for i in range(0, self.nl-1):
        self.b[i] = np.zeros((layers[i+1], 1))
        self.w[i] = np.random.rand(layers[i+1], layers[i])
        self.z[i] = np.zeros((layers[i+1], 1))
        self.a[i] = np.zeros((layers[i+1], 1))

    def __repr__(self):
      s = 'Weight Parameters:\n'
      for k, v in self.w.items():
        s += 'Layer: ' + str(k) + ' , Shape: ' + str(v.shape) + '\n' + str(v) + '\n'

      s += '\nBias Parameters:\n'
      for k, v in self.b.items():
        s += 'Layer: ' + str(k) + ' , Shape: ' + str(v.shape) + '\n' +  str(v) + '\n'

      s += '\nActivations Parameters:\n'
      for k, v in self.a.items():
        s += 'Layer: ' + str(k) + ' , Shape: ' + str(v.shape) + '\n' +  str(v) + '\n'
      return s

    def forward(self, X):
      self.a[0] = np.copy(X)
      for i in range(1, self.nl):
        # TODO: Implement regularization
          # Dropout
          # Frobenius L2
        self.z[i] = np.dot(self.w[i-1], self.a[i-1])
        self.a[i] = relu(self.z[i])

    def backward(self, Y):
      pass

    def learn(self, X, Y, iters=1000):
      # TODO: Implement minibatch
      for i in range(0, iters):
        self.forward(X)
        self.backward(Y)
      pass
