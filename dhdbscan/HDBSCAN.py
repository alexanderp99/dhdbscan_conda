import scipy.linalg
from sklearn.metrics import pairwise_distances
import numpy as np
from itertools import groupby, tee

import scipy as sp

from hdbscan._hdbscan_linkage import (
    single_linkage,
    mst_linkage_core,
    mst_linkage_core_vector,
    label,
)
from numpy import isclose

from hdbscan._hdbscan_tree import (
    condense_tree,
    compute_stability,
    get_clusters,
    outlier_scores,
)
from hdbscan._hdbscan_reachability import mutual_reachability, sparse_mutual_reachability

import numpy as np
from n_tree import run_dhdbscan_from_mst


def sort_deterministically(arr):
    # Find the unique values in the third column and their start indices
    _, idx_start, count = np.unique(arr[:, 2], return_counts=True, return_index=True)
    # This array will store the sorted array
    sorted_arr = np.empty_like(arr)

    # Initialize the position in the sorted array
    pos = 0
    for start, num in zip(idx_start, count):
        # If more than one row shares the third column value, sort these rows
        if num > 1:
            # Extract the rows that need sorting
            sub_arr = arr[start:start + num]
            # Sort these rows first by the first column, then by the second column
            indices = np.lexsort((sub_arr[:, 1], sub_arr[:, 0]))
            # Place sorted rows in the resulting array
            sorted_arr[pos:pos + num] = sub_arr[indices]
        else:
            # If the group has only one row, just copy it to the sorted array
            sorted_arr[pos:pos + num] = arr[start:start + num]
        # Update the position
        pos += num

    return sorted_arr

import numpy as np


class HDBSCAN:
    def __init__(self, min_points=15):
        self.mutual_reachability = None
        self.single_linkage_tree = None
        self.minimum_spanning_tree = None
        self.stabilities = None
        self.probabilities = None
        self.labels_ = None
        self.stability_dict = None
        self.category_dict = None
        self.condensed_tree = None
        self.distance_matrix = None
        self.metric = "euclidean"
        self.p = 2
        self.min_samples = min_points
        self.alpha = 1
        self.min_cluster_size = min_points
        return

    def fit_hdbscan(self, X):

        self.distance_matrix = pairwise_distances(X, metric=self.metric)
        self.mutual_reachability = self.calculate_mutual_reachability(self.distance_matrix, self.min_samples,
                                                                 self.alpha)
        self.mutual_reachability = self.make_symmetric(self.mutual_reachability)

        self.mutual_reachability = np.round(self.mutual_reachability,decimals=6)

        self.minimum_spanning_tree = mst_linkage_core(self.mutual_reachability)
        self.minimum_spanning_tree = self.minimum_spanning_tree[np.argsort(self.minimum_spanning_tree.T[2]), :] # sorting is required! Otherwise trash results
        self.minimum_spanning_tree.T[2] = self.minimum_spanning_tree.T[2].round(decimals=2)

        self.single_linkage_tree = label(self.minimum_spanning_tree)

        self.condensed_tree = condense_tree(self.single_linkage_tree, self.min_cluster_size)

        self.stability_dict = compute_stability(self.condensed_tree)

        self.labels_, self.probabilities, self.stabilities = get_clusters(
            self.condensed_tree,
            self.stability_dict,
            'eom',
            False,
            False,
            0.0,
            0,
        )

        return self

    def fit_hdbscan_distance_matrix(self, X):

        self.distance_matrix = X
        self.mutual_reachability = self.calculate_mutual_reachability(self.distance_matrix, self.min_samples,
                                                                      self.alpha)
        self.mutual_reachability = self.make_symmetric(self.mutual_reachability)

        self.minimum_spanning_tree = mst_linkage_core(self.mutual_reachability)
        self.minimum_spanning_tree = self.minimum_spanning_tree[np.argsort(self.minimum_spanning_tree.T[2]),
                                     :]  # sorting is required! Otherwise trash results

        self.single_linkage_tree = label(self.minimum_spanning_tree)

        self.condensed_tree = condense_tree(self.single_linkage_tree, self.min_cluster_size)

        self.stability_dict = compute_stability(self.condensed_tree)

        self.labels_, self.probabilities, self.stabilities = get_clusters(
            self.condensed_tree,
            self.stability_dict,
            'eom',
            False,
            False,
            0.0,
            0,
        )

        return self

    def fit_hdbscan_mutual_reachability(self, X):

        self.mutual_reachability = X
        self.mutual_reachability = self.make_symmetric(self.mutual_reachability)

        self.minimum_spanning_tree = mst_linkage_core(self.mutual_reachability)
        self.minimum_spanning_tree = self.minimum_spanning_tree[np.argsort(self.minimum_spanning_tree.T[2]),
                                     :]  # sorting is required! Otherwise trash results

        self.single_linkage_tree = label(self.minimum_spanning_tree)

        self.condensed_tree = condense_tree(self.single_linkage_tree, self.min_cluster_size)

        self.stability_dict = compute_stability(self.condensed_tree)

        self.labels_, self.probabilities, self.stabilities = get_clusters(
            self.condensed_tree,
            self.stability_dict,
            'eom',
            False,
            False,
            0.0,
            0,
        )

        return self

    def fit_hdbscan_linkage_matrix(self, X):

        self.minimum_spanning_tree = X
        self.minimum_spanning_tree = self.minimum_spanning_tree[np.argsort(self.minimum_spanning_tree.T[2]),
                                     :]  # sorting is required! Otherwise trash results

        self.single_linkage_tree = self.label(self.minimum_spanning_tree)

        self.category_dict = self.count_categories(self.single_linkage_tree)

        self.condensed_tree = condense_tree(self.single_linkage_tree, self.min_cluster_size)

        self.stability_dict = compute_stability(self.condensed_tree)

        self.labels_, self.probabilities, self.stabilities = get_clusters(
            self.condensed_tree,
            self.stability_dict,
            'eom',
            False,
            False,
            0.0,
            0,
        )

        return self


    def make_symmetric(self,matrix):
        # Get the minimum values from the upper and lower triangular parts
        minimum_triangular_matrix = np.minimum(np.triu(matrix), np.tril(matrix).T)

        # Since numpy fill_diagonal modifies the array in-place, create a symmetric matrix by adding the transpose
        symmetric_m = minimum_triangular_matrix + minimum_triangular_matrix.T

        # Correct the diagonal: The original diagonal should be set correctly.
        # np.fill_diagonal explicitly sets the diagonal and works in-place
        np.fill_diagonal(symmetric_m, np.diagonal(matrix))

        "other idea: np.maximum(matrix,matrix.T)"

        return symmetric_m


    def count_categories(self, linkage_tree):
        unique_items, counts = np.unique(linkage_tree[:, 3], return_counts=True)
        return dict(zip(unique_items, counts))

    def fit_mst(self, X):

        self.minimum_spanning_tree = X

        self.single_linkage_tree = self.label(self.minimum_spanning_tree)

        self.category_dict = self.count_categories(self.single_linkage_tree)

        self.condensed_tree = condense_tree(self.single_linkage_tree, self.min_cluster_size)

        self.stability_dict = compute_stability(self.condensed_tree)

        self.labels_, self.probabilities, self.stabilities = get_clusters(
            self.condensed_tree,
            self.stability_dict,
            'eom',
            False,
            False,
            0.0,
            0,
        )
        return self

    def fit_hdbscan_original(self, X):

        self.distance_matrix = pairwise_distances(X, metric=self.metric)
        self.mutual_reachability = self.calculate_mutual_reachability(self.distance_matrix, self.min_samples,
                                                                      self.alpha)

        self.minimum_spanning_tree = mst_linkage_core(self.mutual_reachability)
        self.minimum_spanning_tree = self.minimum_spanning_tree[np.argsort(self.minimum_spanning_tree.T[2]),
                                     :]  # sorting is required! Otherwise trash results

        self.single_linkage_tree = label(self.minimum_spanning_tree)

        self.condensed_tree = condense_tree(self.single_linkage_tree, self.min_cluster_size)

        self.stability_dict = compute_stability(self.condensed_tree)

        self.labels_, self.probabilities, self.stabilities = get_clusters(
            self.condensed_tree,
            self.stability_dict,
            'eom',
            False,
            False,
            0.0,
            0,
        )
        return self


    def calculate_mutual_reachability(self, distance_matrix, min_points=5, alpha=1.0):
        """Compute the weighted adjacency matrix of the mutual reachability
        graph of a distance matrix.

        Parameters
        ----------
        distance_matrix : ndarray, shape (n_samples, n_samples)
            Array of distances between samples.

        min_points : int, optional (default=5)
            The number of points in a neighbourhood for a point to be considered
            a core point.

        Returns
        -------
        mututal_reachability: ndarray, shape (n_samples, n_samples)
            Weighted adjacency matrix of the mutual reachability graph.

        References
        ----------
        .. [1] Campello, R. J., Moulavi, D., & Sander, J. (2013, April).
           Density-based clustering based on hierarchical density estimates.
           In Pacific-Asia Conference on Knowledge Discovery and Data Mining
           (pp. 160-172). Springer Berlin Heidelberg.
        """
        size = distance_matrix.shape[0]
        min_points = min(size - 1, min_points)
        try:
            core_distances = np.partition(distance_matrix,
                                          min_points,
                                          axis=0)[min_points]
        except AttributeError:
            core_distances = np.sort(distance_matrix,
                                     axis=0)[min_points]

        if alpha != 1.0:
            distance_matrix = distance_matrix / alpha

        stage1 = np.where(core_distances > distance_matrix,
                          core_distances, distance_matrix)
        result = np.where(core_distances > stage1.T,
                          core_distances.T, stage1.T).T
        return result

    def mst_linkage_core(self, distance_matrix):
        n_samples = distance_matrix.shape[0]

        result = np.zeros((n_samples - 1, 3))
        node_labels = np.arange(n_samples, dtype=np.intp)
        current_node = 0
        current_distances = np.full(n_samples, np.inf)
        current_labels = node_labels

        for i in range(1, n_samples):
            label_filter = current_labels != current_node
            current_labels = current_labels[label_filter]
            left = current_distances[label_filter]
            right = distance_matrix[current_node][current_labels]
            current_distances = np.minimum(left, right)

            new_node_index = np.argmin(current_distances)
            new_node = current_labels[new_node_index]
            result[i - 1, 0] = current_node
            result[i - 1, 1] = new_node
            result[i - 1, 2] = current_distances[new_node_index]
            current_node = new_node

        return result
