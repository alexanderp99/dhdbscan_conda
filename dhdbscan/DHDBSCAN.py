import scipy.linalg
from sklearn.metrics import pairwise_distances
import numpy as np

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

class UnionFind:
    def __init__(self, N):
        self.parent_arr = -1 * np.ones(2 * N - 1, dtype=int)
        self.next_label = N
        self.size_arr = np.hstack((np.ones(N, dtype=int), np.zeros(N - 1, dtype=int)))

    def union(self, m, n):
        self.size_arr[self.next_label] = self.size_arr[m] + self.size_arr[n]
        self.parent_arr[m] = self.next_label
        self.parent_arr[n] = self.next_label
        self.size_arr[self.next_label] = self.size_arr[m] + self.size_arr[n]
        self.next_label += 1

    def fast_find(self, n):
        p = n
        while self.parent_arr[n] != -1:
            n = self.parent_arr[n]
        # Path compression
        while self.parent_arr[p] != n:
            p, self.parent_arr[p] = self.parent_arr[p], n
        return n


def label_(L):
    result_arr = np.zeros((L.shape[0], 4))
    N = L.shape[0] + 1
    U = UnionFind(N)

    for index in range(L.shape[0]):
        a = int(L[index, 0])
        b = int(L[index, 1])
        delta = L[index, 2]

        aa = U.fast_find(a)
        bb = U.fast_find(b)

        result_arr[index, 0] = aa
        result_arr[index, 1] = bb
        result_arr[index, 2] = delta
        result_arr[index, 3] = U.size_arr[aa] + U.size_arr[bb]


        U.union(aa, bb)

    return result_arr



class DHDBSCAN:
    def __init__(self):
        self.distance_matrix = None
        self.metric = "euclidean"
        self.p = 2
        self.min_samples = 15
        self.alpha = 1
        self.min_cluster_size = 15
        return

    def fit_deterministic(self, X, y=None):
        pass

    def construct_minimum_spanning_tree(self, X):
        # Convert input matrix to high precision
        X = np.array(X, dtype=np.float128) if hasattr(np, 'float128') else np.array(X, dtype=np.float64)

        if X.shape[0] != X.shape[1]:
            raise ValueError("X needs to be square matrix of edge weights")

        n_vertices = X.shape[0]
        spanning_edges = []
        visited_vertices = [0]
        num_visited = 1

        diag_indices = np.arange(n_vertices)
        X[diag_indices, diag_indices] = np.inf  # Use high precision inf

        while num_visited != n_vertices:
            new_edge = np.argmin(X[visited_vertices], axis=None)
            new_edge = divmod(new_edge, n_vertices)
            new_edge = [visited_vertices[new_edge[0]], new_edge[1]]
            weight = X[new_edge[0], new_edge[1]]
            spanning_edges.append(new_edge + [weight])

            visited_vertices.append(new_edge[1])
            X[visited_vertices, new_edge[1]] = np.inf
            X[new_edge[1], visited_vertices] = np.inf
            num_visited += 1

        minimum_spanning_tree = np.array(spanning_edges, dtype=np.float64)  # Convert back to standard precision
        return minimum_spanning_tree[np.argsort(minimum_spanning_tree[:, 2])]

    def compute_high_precision_difference(self, mutual_reachability):
        # Convert the mutual_reachability matrix to high precision
        mutual_reachability = np.array(mutual_reachability, dtype=np.longdouble)

        # Compute the upper triangular matrix, transpose it, and compute the difference
        result = np.triu(mutual_reachability).T - mutual_reachability

        # Optionally convert back to float64 if reduced precision is acceptable for output
        # result = np.array(result, dtype=np.float64)

        return result

    def make_symmetric_old(self, matrix):
        """
        Did not work as planned, because if elements are shuffled, the triangular matrices may differ, leading to different results still
        :param matrix:
        :return:
        """
        symmetric_m = np.triu(matrix) + np.triu(matrix,k=1).T
        if not scipy.linalg.issymmetric(symmetric_m):
            raise Exception("Matrix is not symmetric")

        return symmetric_m

    def make_symmetric(self,matrix):
        # Get the minimum values from the upper and lower triangular parts
        minimum_triangular_matrix = np.minimum(np.triu(matrix), np.tril(matrix).T)

        # Since numpy fill_diagonal modifies the array in-place, create a symmetric matrix by adding the transpose
        symmetric_m = minimum_triangular_matrix + minimum_triangular_matrix.T

        # Correct the diagonal: The original diagonal should be set correctly.
        # np.fill_diagonal explicitly sets the diagonal and works in-place
        np.fill_diagonal(symmetric_m, np.diagonal(matrix))

        return symmetric_m

    def check_for_determinism(self, input):
        np.random.seed(42)
        shuffled_indices = np.random.permutation(len(input))
        shuffled_data = input[shuffled_indices]

        previous_category_dict = None
        for i in range(10):
            single_linkage_tree = self.label(shuffled_data)

            unique_items, counts = np.unique(single_linkage_tree[:, 3], return_counts=True)
            category_dict = dict(zip(unique_items, counts))
            if previous_category_dict is None:
                previous_category_dict = category_dict
            elif previous_category_dict != category_dict:
                raise Exception("Permutation check failed")


    def count_categories(self, linkage_tree):
        unique_items, counts = np.unique(linkage_tree[:, 3], return_counts=True)
        return dict(zip(unique_items, counts))

    def fit(self, X, y=None):
        self.distance_matrix = pairwise_distances(X, metric=self.metric)

        self.mutual_reachability = self.calculate_mutual_reachability(self.distance_matrix, self.min_samples, self.alpha)
        self.mutual_reachability = self.make_symmetric(self.mutual_reachability)

        self.minimum_spanning_tree = self.construct_minimum_spanning_tree(self.mutual_reachability)
        self.minimum_spanning_tree = self.mst_linkage_core(self.mutual_reachability)
        self.minimum_spanning_tree = self.minimum_spanning_tree[np.argsort(self.minimum_spanning_tree.T[2]), :]
        #self.minimum_spanning_tree = sort_deterministically(self.minimum_spanning_tree)

        self.check_for_determinism(self.minimum_spanning_tree)

        ## Version 1. Custom implementation
        self.single_linkage_tree = label_(self.minimum_spanning_tree)
        ## Version 2. C++ implementation
        #self.single_linkage_tree = self.label(self.minimum_spanning_tree)
        #all(np.equal(self.label(self.minimum_spanning_tree)[:,3],label_(self.minimum_spanning_tree)[:,3]))


        #self.label(np.array([[1,2,0.1],[2,3,0.2],[3,4,0.3]]))

        #label_(np.array([[1, 2, 0.1], [4, 3, 0.2], [4, 1, 0.4]]))

        #not deterministic:
        # self.label(np.array([[1,2,0.1],[4,3,0.2],[2,3,0.2],[4,5,0.3]]))
        # self.label(np.array([[1,2,0.1],[2,3,0.2],[4,3,0.2],[4,5,0.3]]))

        ## label_(np.array([[1,2,0.1],[4,3,0.2],[2,3,0.2],[4,5,0.3]]))
        ## label_(np.array([[1,2,0.1],[2,3,0.2],[4,3,0.2],[4,5,0.3]]))

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

    def label(self, minimum_spanning_tree):
        return label(minimum_spanning_tree)

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
        distance_matrix = np.array(distance_matrix)

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