import numpy as np

class NeuralNetwork:
    def __init__(self, layers, rseed=0):
      np.random.seed(rseed)
      self.e = 0.1
      self.a = {}
      self.z = {}
      self.b = {}
      self.w = {}

      # append an extra layer here
      # will need to change this in the event that we are doing multi-class classification
      layers = layers + [1]
      self.nl = len(layers)
      for i in range(0, self.nl-1):
        self.b[i] = np.zeros((layers[i+1], 1))
        self.w[i] = np.random.rand(layers[i], layers[i+1])

    def __repr__(self):
      s = 'Weight Parameters:\n'
      for k, v in self.w.items():
        s += 'Layer: ' + str(k) + ' , Shape: ' + str(v.shape) + '\n' + str(v) + '\n'

      s += '\nBias Parameters:\n'
      for k, v in self.b.items():
        s += 'Layer: ' + str(k) + ' , Shape: ' + str(v.shape) + '\n' +  str(v) + '\n'
      return s


    def forward(self, A0):
      for i in range(0, self.nl):
        pass

    def back(self, A0):
      pass

    def train(self, X):
      pass

if __name__ == '__main__':
    nn = NeuralNetwork(layers=[8, 6, 4], rseed=32)
    a = np.random.rand(3, 100)
    w = np.random.rand(8, 3)
    b = np.zeros([8, 1])
    z = np.dot(w, a)  + b
    print(z.shape)
