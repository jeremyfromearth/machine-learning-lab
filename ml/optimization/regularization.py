class PassThruRegularization:
    def step(self, m, η):
        return 1.0
    
class L2Regularization:
    def __init__(self):
        self.λ = 0.001
    
    def step(self, m, η):
        return 1 - (η * (self.λ / m))
