import numpy as np
from dhdbscan.HDBSCAN import HDBSCAN
from scipy.linalg import issymmetric
np.random.seed(42)

l = []
l.append(np.array([[0, 1, 0.1], [3, 2, 0.2], [1, 2, 0.2], [3, 4, 0.3]]))
l.append(np.array([[0, 1, 0.1], [1, 2, 0.2], [3, 2, 0.2], [3, 4, 0.3]]))

"""
Todo
"""

for each_l in l:
    clusterer = HDBSCAN(min_points=2)
    clusterer.fit_hdbscan_linkage_matrix(each_l)
    print("Dataset")
    print(each_l)
    print("Single Linkage tree")
    print(clusterer.single_linkage_tree)
    print("\n")

