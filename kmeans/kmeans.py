from random import randrange as rand
from operator import itemgetter
import numpy as np
import sys

class KMeansModel(object):
    '''
    K-Means model
    '''
    def __init__(self, data):
        self.data = data
        self.means = []
        self.clusters = {}
        self.iterations = 0
        self.max_iterations = 1024
        self.maxX = -float('inf')
        self.maxY = -float('inf')
        self.minX = float('inf')
        self.minY = float('inf')
        for v in self.data:
            x = float(v[0])
            y = float(v[1])
            self.minX = min(x, self. minX)
            self.minY = min(y, self. minY)
            self.maxX = max(x, self.maxX)
            self.maxY = max(y, self.maxY)

    def run(self, k):
        convergence = False
        self.iterations = 0
        self.means = [np.array([rand(self.minX, self.maxX), rand(self.minY, self.maxY)]) for x in range(k)]
        while convergence is False:
            self.iterations += 1
            self.clusters = {i : [] for i in range(0,k)}
            for v in self.data:
                cluster = -1
                distance = float('inf')
                for i in range(len(self.means)):
                    d = np.linalg.norm(v - self.means[i])
                    if d < distance:
                        distance = d
                        cluster = i
                self.clusters[cluster].append(v)

            # Create a list to store the new means
            new_means = []        
            for key in self.clusters:
                # Sum the vectors in each cluster
                summed = np.zeros(2, dtype=np.float)
                cluster = self.clusters[key]
                for v in cluster:
                    summed += v
               
                # Find the mean of all the vectors in each cluster
                count = len(cluster)
                if count > 0:
                    new_means.append(summed / np.array([count, count]))
                else:
                    new_means.append(self.means[key])
            
            # If the new means are the same as the old, the algorithm has converged
            convergence = np.array_equal(self.means, new_means)
            self.means = new_means
