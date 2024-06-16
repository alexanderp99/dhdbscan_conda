from collections import defaultdict

import numpy as np
from dhdbscan.HDBSCAN import HDBSCAN
import matplotlib.pyplot as plt
from dhdbscan.HDBSCAN import HDBSCAN as MyHdbscan


rows = 6
columns = 6
#ones_matrix_1 = np.ones((rows, columns))
#ones_matrix_2 = np.ones((rows, columns))*2
#ones_matrix_3 = np.ones((rows, columns))*3

ones_matrix_1 = np.random.uniform(1, 1, size=(rows, columns))
ones_matrix_2 = np.random.uniform(1, 1, size=(rows, columns))
ones_matrix_3 = np.random.uniform(1, 1, size=(rows, columns))

lower_bound = 7
upper_bound = 15
big_distance_matrix = np.random.uniform(lower_bound, upper_bound, size=(rows * 3, columns * 3))

big_distance_matrix[:rows, :columns] = ones_matrix_1
big_distance_matrix[rows:2*rows, columns:2*columns] = ones_matrix_2
big_distance_matrix[2*rows:, 2*columns:] = ones_matrix_3
np.fill_diagonal(big_distance_matrix,0)

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

datasets = {
    'dataset1': {'dataset': big_distance_matrix, 'min_cluster_size': 4}
}
label_dicts = []

shuffle_corrected_labels = []
shuffle_corrected_probabilities = []

for dataset_name, each_datadict in datasets.items():
    each_dataset = each_datadict['dataset']
    min_cluster_size = each_datadict['min_cluster_size']

    for i in range(5):
        shuffled_indices = np.random.permutation(len(each_dataset))
        shuffled_data = each_dataset.copy()  # Make a copy to preserve the original dataset
        shuffled_data = shuffled_data[shuffled_indices]

        clusterer = MyHdbscan(min_points=min_cluster_size)
        clusterer.fit_hdbscan_distance_matrix(shuffled_data)

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
        shuffle_corrected_labels.append(labelz)

print("All labels are equal:", check_identical_arrays(shuffle_corrected_labels))
print("All label counts are equal", check_identical_arrays(label_dicts))
