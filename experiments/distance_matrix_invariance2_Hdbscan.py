from collections import defaultdict

import numpy as np
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt
from dhdbscan.HDBSCAN import HDBSCAN as MyHdbscan

def check_identical_arrays(arrays):
    first_array = arrays[0]
    for array in arrays[1:]:
        if not np.array_equal(first_array, array):
            diff_indices = np.where(first_array != array)[0]
            print(f"Arrays differ at indices: {diff_indices}")
            return False
    return True


"""
Checks if shuffeling a distance matrix leads to different labels. The distance matrix has points with same distance.
"""

def check_identical_labels(arrays):
    first_array = arrays[0]

    equal = False

    for each_array in arrays[1:]:
        unique1 = np.unique(first_array)
        unique2 = np.unique(each_array)

        if len(unique1) != len(unique2):
            equal = False
            break
        else:
            mapping = {u1: u2 for u1, u2 in zip(unique1, unique2)}
            mapped_array1 = np.vectorize(mapping.get)(first_array)
            equal = np.array_equal(mapped_array1, each_array)
            if not equal:
                print(f"Not equal: {mapped_array1, each_array}")

    return equal

def generate_n_simplex_with_radius(n, starting_point, radius):
    # Lambda function to generate initial n-simplex vertices
    vertices = lambda n: [i*[0]+[1]+(n-1-i)*[0] for i in range(n)] + [[1/np.sqrt(n)]*n]

    # Convert to numpy array
    simplex_vertices = np.array(vertices(n))

    # Center the simplex vertices
    centroid = np.mean(simplex_vertices, axis=0)
    simplex_vertices -= centroid

    # Scale vertices to match the specified radius
    scale_factor = radius / np.linalg.norm(simplex_vertices[0])
    simplex_vertices *= scale_factor

    # Translate vertices to the starting point
    starting_point = np.array(starting_point).reshape(1, -1)
    simplex_vertices += starting_point

    return simplex_vertices


# Example usage:
n = 2  # Dimension of the simplex (n-simplex in n+1 dimensions)
#datapoints = [generate_n_simplex_with_starting_point(n, np.array([0, 0, 0])),generate_n_simplex_with_starting_point(n, np.array([-5, 0, 0])),generate_n_simplex_with_starting_point(n, np.array([5, 0, 0]))]
datapoints = np.vstack([generate_n_simplex_with_radius(n, np.array([0, 0]),1),generate_n_simplex_with_radius(n, np.array([-5,-5]),3),generate_n_simplex_with_radius(n, np.array([5,5]),2)])
datapoints = np.array([[-0.5,0],[0.5,0],[0,np.sqrt(3)/2],[19,0],[20,0],[19.5,np.sqrt(3)/2],[9.75,-18.5*np.sqrt(3)/2],[28.25,-16],[-15,-40],[-10,-40],[-15,-35],[-10,-35]])
datapoints = np.array([[-0.5,0],[0.5,0],[0,np.sqrt(3)/2],[19,0],[20,0],[19.5,np.sqrt(3)/2],[9.75,-18.5*np.sqrt(3)/2],[-15,-40],[-10,-40],[-15,-35],[-10,-35]])


datasets = {
    'dataset1': {'dataset': datapoints, 'min_cluster_size': 2}
}
label_dicts = []

shuffle_corrected_labels = []
shuffle_corrected_probabilities = []

for dataset_name, each_datadict in datasets.items():
    each_dataset = each_datadict['dataset']
    min_cluster_size = each_datadict['min_cluster_size']

    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True, approx_min_span_tree=False,
                        gen_min_span_tree=True, algorithm="generic", metric="euclidean")
    # clusterer = MyHdbscan(min_points=min_cluster_size)
    # clusterer.fit_hdbscan_distance_matrix(shuffled_data)
    # clusterer.fit_hdbscan(shuffled_data)
    clusterer.fit(each_dataset)
    #shuffle_corrected_labels.append(clusterer.labels_)

    for i in range(5):
        shuffled_indices = np.random.permutation(len(each_dataset))
        #shuffled_indices = np.arange(len(each_dataset))
        shuffled_data = each_dataset.copy()  # Make a copy to preserve the original dataset
        shuffled_data = shuffled_data[shuffled_indices]

        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True, approx_min_span_tree=False,
                        gen_min_span_tree=True, algorithm="generic", metric="euclidean")
        #clusterer = MyHdbscan(min_points=min_cluster_size)
        #clusterer.fit_hdbscan_distance_matrix(shuffled_data)
        #clusterer.fit_hdbscan(shuffled_data)
        clusterer.fit(shuffled_data)

        """clusterer.condensed_tree_.plot(select_clusters=True)
        plt.title("Hdbscan condensed tree")
        plt.show()"""

        labels = clusterer.labels_[shuffled_indices]
        probabilities = clusterer.probabilities_

        cop = np.copy(clusterer.labels_)
        cop[shuffled_indices] = clusterer.labels_
        shuffle_corrected_labels.append(cop)

        labels = cop

        num_colors = len(set(labels))
        palette = plt.cm.viridis(np.linspace(0, 1, num_colors))
        cluster_colors = [palette[col] if col >= 0 else (0.5, 0.5, 0.5) for col in labels]
        plt.scatter(each_dataset.T[0], each_dataset.T[1], c=cluster_colors)
        plt.title("Hdbscan scatterplot matplotlib")
        plt.show()

        clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
        plt.title("Hdbscan single linkage tree")
        plt.show()

        if not np.array_equal(shuffled_data, each_dataset[shuffled_indices]):
            raise Exception("Relabeled rows are not identical")

        labelz = clusterer.labels_[shuffled_indices]
        label_counts = defaultdict(int)
        for label in labelz:
            label_counts[label] += 1
        label_counts_dict = dict(label_counts)
        sorted_label_counts = dict(sorted(label_counts_dict.items()))
        label_count_ndarray = np.array([item[1] for item in sorted(label_counts.items())])

        label_dicts.append(label_count_ndarray)


print("All labels are equal:", check_identical_labels(shuffle_corrected_labels))

