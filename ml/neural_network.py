import numpy as np
from kernels import sigmoid, relu

class FeedForwardBinaryNet(object):
  '''
  Two layer neural network using relu activations and sigmoid binary output
  '''
  def __init__(self, width):
    '''
      width - Number of neurons in the hidden layer
    '''
    self.width = width
    self.loss_history = []
    self.learning_rate = 0.1

  def forward(self, X):
    '''
      Make a single forward pass over the data
      Compute activations
    '''
    z1 = np.add(np.dot(self.w1, X), self.b1)
    a1 = relu(z1)
    z2 = np.add(np.dot(self.w2, a1), self.b2)
    a2 = sigmoid(z2)
    return z1, z2, a1, a2

  def learn(self, X, Y, epochs):
    '''
      X - Training examples as columns
      Y - Training labels
      epochs - Number of epochs to train for
    '''
    # initialize parameters
    n_x = X.shape[0]
    one_over_m = 1 / X.shape[1]
    self.b1 = np.zeros((self.width, 1))
    self.b2 = 0
    self.w1 = np.random.random((self.width, n_x))
    self.w2 = np.random.random((1, self.width))

    
    for i in range(epochs):
      # make a forward pass and calculate the loss
      z1, z2, a1, a2 = self.forward(X)
      loss = np.multiply(one_over_m, np.sum((Y - a2) ** 2).ravel())
      self.loss_history.append(loss)

      # calculate derivatives
      # https://www.youtube.com/watch?v=P7_jFxTtJEo&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=32
      # https://www.youtube.com/watch?v=7bLEWDZng_M&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=33
      dz2 = a2 - Y
      dw2 = one_over_m * np.dot(dz2, a1.T)
      db2 = one_over_m * np.sum(dz2).ravel()
      dz1 = np.copy(z1)
      dz1[dz1 > 0] = 1
      dz1[dz1 < 0] = 0
      dz1 = np.dot(self.w2.T, dz2) * dz1
      dw1 = one_over_m * np.dot(dz1, X.T)
      db1 = one_over_m * np.sum(dz1, axis=1, keepdims=True)
      
      # update parameters
      self.w2 -= dw2 * self.learning_rate
      self.b2 -= db2 * self.learning_rate
      self.w1 -= dw1 * self.learning_rate
      self.b1 -= db1 * self.learning_rate
    print('Final loss', loss)

  def predict(self, x):
    z1, z2, a1, a2 = self.forward(x)
    return np.round(a2)

class FeedForwardMulticlassNet(object):
  pass

class FeedForwardDeepNet(object):
  pass

class ConvolutionalDeepNet(object):
  pass

class RecurrentNet(object):
  pass

class LongShortTermMemoryNet(object):
  pass

if __name__ == '__main__':
  import csv
  np.random.seed(0)
  with open('./datasets/bears.csv') as f:
    r = csv.DictReader(f)
    features = ['HEADLEN', 'HEADWTH', 'NECK', 'LENGTH', 'CHEST', 'WEIGHT']
    X = np.array([[row[i] for i in features] for row in r], dtype='float')
    X = X/X.max(axis=0)
    f.seek(0)
    r = csv.DictReader(f)
    Y = np.array([row['SEX'] for row in r], dtype='float')
    Y[Y == 1] = 0
    Y[Y == 2] = 1
    x_train = X[:30]
    y_train = Y[:30]
    x_test = X[30:]
    y_test = Y[30:]

  ffn = FeedForwardBinaryNet(8)
  ffn.learning_rate = 0.1
  ffn.learn(x_train.T, y_train, 2000)
  prediction = ffn.predict(x_test.T).ravel()
  correct = len(np.where(np.equal(prediction, y_test))[0])
  total = y_test.shape[0]
  accuracy = correct / total

  print('Completed with {} of {} samples classified correctly at an overall accuracy of {}%'
    .format(correct, total, np.round((correct/total)*100, 3)))
