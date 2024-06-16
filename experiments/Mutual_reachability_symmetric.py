import numpy as np
from hdbscan import HDBSCAN
from scipy.linalg import issymmetric
from dhdbscan.HDBSCAN import HDBSCAN as MyHdbscan

dataset = np.load('../clusterable_data.npy')
np.random.seed(42)

"""
Result: 
-The mutual reachability matrix and the distance matrix are not symmetric, due to rounding errors. 
"""

min_cluster_size = 15

for i in range(3):
    shuffled_indices = np.random.permutation(len(dataset))
    shuffled_data = dataset[shuffled_indices]

    clusterer = MyHdbscan(min_points=min_cluster_size)
    clusterer.fit_hdbscan_original(shuffled_data) #the original fit function works like the original HDBSCAN fit
    print("Clusterer Distance Matrix is symmetric:", issymmetric(clusterer.distance_matrix))
    print("Clusterer Mutual Reachability Matrix is symmetric:", issymmetric(clusterer.mutual_reachability))
    print("Clusterer Mutual Reachability Matrix is symmetric with tolerance:", issymmetric(clusterer.mutual_reachability, atol=1e-8))

    my_hdbscan_clusterer = MyHdbscan(min_points=min_cluster_size)
    my_hdbscan_clusterer.fit_hdbscan(shuffled_data) #this function makes the mutual reachability matrix symmetric --> Max of Triangle matrices
    print("my_hdbscan_clusterer Distance Matrix is symmetric:", issymmetric(my_hdbscan_clusterer.distance_matrix))
    print("my_hdbscan_clusterer Mutual Reachability Matrix is symmetric:", issymmetric(my_hdbscan_clusterer.mutual_reachability))
    print("\n")




