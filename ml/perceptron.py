import numpy as np

class Perceptron(object):
    def __init__(self, eta=0.01, iterations = 10):
        self.eta = eta
        self.iterations = iterations
        self.weights = []

    def learn(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])
        self.errors = [];
        
        for i in range(self.iterations):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


if __name__ == '__main__':
    p = Perceptron()
