import sys
import numpy as np
from kmeans import KMeans
from collections import deque
from matplotlib import pyplot as plt
from matplotlib import style

# Clean up the raw data and create the dataset
data = []
data_file = sys.argv[1]
lines = [line.strip('\n') for line in open(data_file) if '#' not in line]

for line in lines:
    parts = line.split()
    if len(parts) == 2:
        x = float(parts[0])
        y = float(parts[1])
        data.append(np.array([x, y]))

# Create a new k-means model and run it
k = int(sys.argv[2])
model = KMeans(data)
model.run(k)

# Create a simple visualization using Matplotlib
x = [data[0] for v in data]
y = [data[1] for v in data]
colors = deque(['b', 'r', 'w', 'k', 'c', 'm', 'y', 'g', 'orange', 'brown'])

for k in model.clusters:
    x = [v[0] for v in model.clusters[k]]
    y = [v[1] for v in model.clusters[k]]
    plt.scatter(x, y, color=colors[0])
    colors.rotate()

for mean in model.means:
    x = [v[0] for v in model.means]
    y = [v[1] for v in model.means]
    plt.scatter(x, y, color='r', s=32, alpha=0.3)

plt.title('K-means Clustering : %d Iterations' % (model.iterations))
plt.ylabel('Y')
plt.xlabel('X')
plt.show()
