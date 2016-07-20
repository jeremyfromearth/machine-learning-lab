import sys
import numpy as np

class KMeansModel(object):
    def __init__(self):
        self.means = [] 
        self.clusters = {}
        self.iterations = 0
        self.quality_history = []
        self.max_iterations = 1024
        self.initializer = ForgyKMeansInitializer()

    def learn(self, dataframe, features, k):
        convergence = False
        self.iterations = 0
        data = dataframe[features].as_matrix()
        self.means = self.initializer.get_initial_means(data, k)
        while convergence is False:
            quality = 0
            self.iterations += 1
            self.clusters = {i : [] for i in range(0, k)}
            for v in data:
                cluster = -1
                distance = float('inf')
                for i in range(len(self.means)):
                    d = np.linalg.norm(v - self.means[i])
                    quality += d
                    if d < distance:
                        distance = d
                        cluster = i
                self.clusters[cluster].append(v)
            self.quality_history.append(quality)

            # Compute the new means
            new_means = []
            for key, cluster in self.clusters.items():
                # Sum the vectors in each cluster
                summed = np.zeros(data.shape[1], dtype=np.float)
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
            
    def predict(self, dataframe, features):
        result = []
        data = dataframe[features].as_matrix()
        for v in data:
            cluster = None
            delta = float('inf')
            for i in range(len(self.means)):
                d = np.linalg.norm(v - self.means[i])
                if d < delta:
                    cluster = i
                    delta = d
            result.append(cluster)
        return result
                
class ForgyKMeansInitializer:
    def get_initial_means(self, data, k):
        random_indices = np.random.randint(0, data.shape[0], k)
        return [data[i] for i in random_indices]

class KMeansPlusPlusInitializer:
    def get_initial_means(self, data, k):
        means = [data[np.random.randint(0, data.shape[0])]]
        while(len(means) < k):
            distance = 0
            new_mean = None
            for d in data:
                for mean in means:
                    new_distance = np.linalg.norm(mean - d) ** 2
                    if new_distance > distance:
                        new_mean = d
                        distance = new_distance 
            means.append(new_mean)
        return means
