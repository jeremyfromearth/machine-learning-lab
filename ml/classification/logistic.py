class LogisticRegressionModel:
    def __init__(self):
        self.params = []
        self.optimization = GradientDescentOptmizer()
        self.regularization = L2Regularization()

    def learn(self, x, y):
        # train the model using the specified optimization
        pass

    def predict(self, x):
        # make a class prediction from the input
        pass

    def compute_cost(self):
        # not necessary for actually training the model, but useful for testing and validation
        pass

