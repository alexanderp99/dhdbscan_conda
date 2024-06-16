import numpy as np
from dhdbscan.HDBSCAN import HDBSCAN
from scipy.linalg import issymmetric

dataset = np.load('../clusterable_data.npy')
np.random.seed(42)

def check_identical_arrays(arrays):
    first_array = arrays[0]
    for array in arrays[1:]:
        if not np.array_equal(first_array, array):
            diff_indices = np.where(first_array != array)[0]
            print(f"Arrays differ at indices: {diff_indices}")
            return False
    return True

def check_mst_weights_equal(arrays):
    first_array = arrays[0]

    for array in arrays[1:]:
        if not np.allclose(first_array[:,2], array[:,2]):
            return False
    return True

"""
Results:
- Hdbscan labels differ if the dataset is shuffled (n points are shuffled)
- If shuffled, the mst stays equal with respect to its weights
"""
number_of_elements_to_shuffle = 100

shuffle_corrected_labels = []
msts = []
mutual_reachabilities = []
categories = []
condensed_trees = []
single_linkage_trees = []

for i in range(5):
    shuffled_indices = np.random.permutation(number_of_elements_to_shuffle)
    shuffled_data = dataset.copy()  # Make a copy to preserve the original dataset
    shuffled_data[:number_of_elements_to_shuffle] = shuffled_data[shuffled_indices]

    clusterer = HDBSCAN(min_points=15)
    clusterer.fit_hdbscan(shuffled_data)
    semi_labels = np.concatenate([shuffled_indices, np.arange(number_of_elements_to_shuffle, len(dataset))])

    msts.append(clusterer.minimum_spanning_tree)
    mutual_reachabilities.append(clusterer.mutual_reachability)
    categories.append(clusterer.category_dict)
    condensed_trees.append(clusterer.condensed_tree)
    single_linkage_trees.append(clusterer.single_linkage_tree)
    shuffle_corrected_labels.append(clusterer.labels_[semi_labels])

print("All labels are equal:", check_identical_arrays(shuffle_corrected_labels))
print("Mst weights are equal", check_mst_weights_equal(msts))
