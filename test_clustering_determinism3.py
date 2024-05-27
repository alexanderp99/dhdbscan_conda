import numpy as np
import matplotlib.pyplot as plt
import os
import sys
#import hdbscan
#sys.path.append("../hdbscan")  # Add the submodule directory to the Python path
#from dhdbscan.DHDBSCAN import DHDBSCAN
#from hdbscan.hdbscan_ import HDBSCAN
from sklearn.base import BaseEstimator, ClusterMixin

from hdbscan import HDBSCAN
from dhdbscan.DHDBSCAN import DHDBSCAN
data = np.load('clusterable_data.npy')
ndarrays = []


"""
This script was used to look if the cluster that the algorithm produces (sorted by number of points), has the same number of clusterpoints
It turns out: Hdbscan is not shuffle deterministic
"""

def get_sorted_cluster_sizes(labels):
    cluster_sizes = np.bincount(labels[labels >= 0])
    return sorted(cluster_sizes, reverse=True)


def test_determinism_num_clusters(data, n=2):
    initial_sorted_sizes = None
    np.random.seed(42)
    msts = []
    mutual_reachabilities = []
    categories = []
    condensed_trees = []
    single_linkage_trees = []

    for _ in range(n):
        shuffled_indices = np.random.permutation(len(data))
        shuffled_data = data[shuffled_indices]
        ## Scenario 1.
        """clusterer = HDBSCAN(min_cluster_size=15, prediction_data=True, approx_min_span_tree=False,
                            gen_min_span_tree=True, algorithm="generic", metric="euclidean").fit(shuffled_data)"""
        ## Scenario 2:
        clusterer = DHDBSCAN().fit(shuffled_data)

        if type(clusterer).__name__ == "DHDBSCAN":

            msts.append(clusterer.minimum_spanning_tree)
            mutual_reachabilities.append(clusterer.mutual_reachability)
            categories.append(clusterer.category_dict)
            condensed_trees.append(clusterer.condensed_tree)
            single_linkage_trees.append(clusterer.single_linkage_tree)

        reversed_labels = np.zeros_like(clusterer.labels_)
        reversed_labels[shuffled_indices] = clusterer.labels_
        current_sorted_sizes = get_sorted_cluster_sizes(reversed_labels)

        if initial_sorted_sizes is None:
            initial_sorted_sizes = current_sorted_sizes
        else:
            if initial_sorted_sizes != current_sorted_sizes:
                return False

    #all_weights_sorted_identical = all(np.equal(msts[0][:,2],msts[1][:,2]))
    #non_equal_row_idx = np.where(np.equal(msts[0][:,2],msts[1][:,2])!=True)
    #all(np.equal(np.unique(msts[0][:,1]) + np.unique(msts[0][:,1]),np.unique(msts[1][:,1]) + np.unique(msts[1][:,1])))

    # sorted_row_entries_mst=np.sort(msts[0][:,0:2],axis=1) -> no help
    # number_of_unique_weights = len(np.unique(msts[0][:, 2]))
    # len(np.unique(msts[0][:,2])), len(np.unique(msts[1][:,2]))
    msts

    return True


def test_determinism_num_clusters_without_shuffeling(data, n=2):
    initial_sorted_sizes = None

    for _ in range(n):
        clusterer = HDBSCAN(min_cluster_size=15, prediction_data=True, approx_min_span_tree=False).fit(data)
        labels = np.copy(clusterer.labels_)
        current_sorted_sizes = get_sorted_cluster_sizes(labels)
        if initial_sorted_sizes is None:
            initial_sorted_sizes = current_sorted_sizes
        else:
            if initial_sorted_sizes != current_sorted_sizes:
                return False
    return True


is_deterministic = test_determinism_num_clusters(data, 3)
print("The algorithm is deterministic" if is_deterministic else "The algorithm is not deterministic.")

