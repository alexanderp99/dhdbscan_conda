import numpy as np
from dhdbscan.HDBSCAN import HDBSCAN
from scipy.linalg import issymmetric
dataset = np.load('../../clusterable_data.npy')
np.random.seed(42)

"""
Checks if the Distance matrix and Mutual reachability are symmetric, as they should be, if the dataset is shuffled.
"""

for i in range(3):
    shuffled_indices = np.random.permutation(len(dataset))
    shuffled_data = dataset[shuffled_indices]
    clusterer = HDBSCAN(min_points=15)
    clusterer.fit_hdbscan(shuffled_data)
    print("Distance Matrix is symmetric:", issymmetric(clusterer.distance_matrix))
    print("Mutual Reachability Matrix is symmetric:", issymmetric(clusterer.mutual_reachability))
    print("\n")
