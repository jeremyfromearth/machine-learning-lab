import numpy as np
from math import pi, sqrt, e

def epanechnikov(x):
    return 3.0/4.0 * (1 - x**2)

def gaussian(x):
    return (1.0 / sqrt(2 * pi)) * (e **-(1/2 * x**2))

def logistic(x):
    return 1.0 / (exp(x) + 2 + exp(-x))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def triweight(x):
    return (35.0/32.0) * (1 - x**2) **2

