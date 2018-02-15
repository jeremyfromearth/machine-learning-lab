import numpy as np
data = [
        {"walk-dog": 1, "weather": "sunny", "wind": "low"},
        {"walk-dog": 1, "weather": "sunny", "wind": "mid"},
        {"walk-dog": 1, "weather": "sunny", "wind": "high"},
        {"walk-dog": 1, "weather": "overcast", "wind": "low"},
        {"walk-dog": 1, "weather": "overcast", "wind": "mid"},
        {"walk-dog": 0, "weather": "overcast", "wind": "high"},
        {"walk-dog": 0, "weather": "rainy", "wind": "high"},
        {"walk-dog": 0, "weather": "rainy", "wind": "mid"},
        {"walk-dog": 0, "weather": "rainy", "wind": "low"},
    ]

class DecisionTree(dict):
    def learn(self, x, y):
        pass

    def predict(self, x):
        pass
