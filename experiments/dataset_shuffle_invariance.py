import numpy as np
from dhdbscan.HDBSCAN import HDBSCAN
from scipy.linalg import issymmetric

dataset = np.load('../clusterable_data.npy')
np.random.seed(42)

def check_identical_arrays(arrays):
    first_array = arrays[0]
    for array in arrays[1:]:
        if not np.array_equal(first_array, array):
            return False
    return True

"""
Result: Hdbscan returns the same labels, if NOT shuffled
"""

shuffle_corrected_labels = []
for i in range(3):
    clusterer = HDBSCAN(min_points=15)
    clusterer.fit_hdbscan(dataset)
    shuffle_corrected_labels.append(clusterer.labels_)

print("All labels are equal:", check_identical_arrays(shuffle_corrected_labels))

