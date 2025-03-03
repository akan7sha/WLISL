import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import matplotlib as mpl

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

mpl.rcParams['figure.dpi'] = 300
savefig_options = dict(format = 'png',dpi=300, bbox_inches = "tight")
def compute_euclidean_distance_matrix(x, y) -> np.array:
    """Calculate distance matrix
    This method calcualtes the pairwise Euclidean distance between two sequences.
    The sequences can have different lengths.
    """
    dist = np.zeros((len(y), len(x)))
    for i in range(len(y)):
        for j in range(len(x)):
            dist[i,j] = (x[j]-y[i])**2
    return dist


def compute_accumulated_cost_matrix(x, y) -> np.array:
    """Compute accumulated cost matrix for warp path using Euclidean distance
    """
    distances = compute_euclidean_distance_matrix(x, y)

    # Initialization
    cost = np.zeros((len(y), len(x)))
    cost[0, 0] = distances[0, 0]

    for i in range(1, len(y)):
        cost[i, 0] = distances[i, 0] + cost[i - 1, 0]

    for j in range(1, len(x)):
        cost[0, j] = distances[0, j] + cost[0, j - 1]

        # Accumulated warp path cost
    for i in range(1, len(y)):
        for j in range(1, len(x)):
            cost[i, j] = min(
                cost[i - 1, j],  # insertion
                cost[i, j - 1],  # deletion
                cost[i - 1, j - 1]  # match
            ) + distances[i, j]

    return cost


# Create two sequences
time1 = np.linspace(start=0, stop=1, num=60)
time2 = time1[0:80]

x1 = 3 * np.sin(np.pi * time1) + 1.5 * np.sin(4*np.pi * time1)
x2 = 3 * np.sin(np.pi * time2 + 0.5) + 1.5 * np.sin(4*np.pi * time2 + 0.5)
distance, warp_path = fastdtw(x1, x2, dist=euclidean)
fig, ax = plt.subplots(figsize=(16, 12))

# Remove the border and axes ticks
fig.patch.set_visible(False)
ax.axis('off')

for [map_x, map_y] in warp_path:
    ax.plot([map_x, map_y], [x1[map_x], x2[map_y]], '-k')

ax.plot(x1, color='blue', marker='o', markersize=10, linewidth=5)
ax.plot(x2, color='red', marker='o', markersize=10, linewidth=5)
ax.tick_params(axis="both", which="major", labelsize=18)

fig.savefig("ex2_dtw_distance1.png", **savefig_options)
