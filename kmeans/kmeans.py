import sys
import numpy as np

class KMeansModel(object):
    def __init__(self):
        self.means = []
        self.clusters = {}
        self.iterations = 0
        self.max_iterations = 1024
        self.initializer = ForgyKmeansInitializer()

    def learn(self, data, k):
        convergence = False
        self.iterations = 0
        self.means = self.initializer.get_initial_means(data, k)
        while convergence is False:
            self.iterations += 1
            self.clusters = {i : [] for i in range(0, k)}
            for v in data:
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
                summed = np.zeros(data.shape[1], dtype=np.float)
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
            if self.iterations >= self.max_iterations or np.array_equal(self.means, new_means):
                convergence = True
            self.means = new_means

class ForgyKMeansInitializer:
    def get_initial_means(self, data, k):
        random_indices = np.random.randint(0, data.shape[0], k)
        return [data[i] for i in random_indices]


