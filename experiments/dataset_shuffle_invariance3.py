import numpy as np
from hdbscan import HDBSCAN
from dhdbscan.HDBSCAN import HDBSCAN as MyHdbscan
from scipy.linalg import issymmetric
from dense_generator import generate_density_data
from collections import defaultdict

def create_datasets(data_generators):
    datasets = {}
    for name, (generator, min_cluster_size) in data_generators.items():
        x, y = generator()
        dataset = np.hstack((x, y))  # Merging x and y columns
        datasets[name] = {'dataset': dataset, 'min_cluster_size': min_cluster_size}
    return datasets


data_generators = {
    "density": (generate_density_data, 10)
}

datasets = create_datasets(data_generators)

np.random.seed(42)

def check_identical_arrays(arrays):
    first_array = arrays[0]
    for array in arrays[1:]:
        if not np.array_equal(first_array, array):
            diff_indices = np.where(first_array != array)[0]
            print(f"Arrays differ at indices: {diff_indices}")
            return False
    return True

"""
Result: If we shuffle the data, even 5 datapoints, the labels change, and even multiple probabilities
"""
number_of_elements_to_shuffle = 5
label_dicts = []

shuffle_corrected_labels = []
shuffle_corrected_probabilities = []
for dataset_name, each_datadict in datasets.items():
    each_dataset = each_datadict['dataset']
    min_cluster_size = each_datadict['min_cluster_size']

    """clusterer = HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True, approx_min_span_tree=False,
                        gen_min_span_tree=True, algorithm="generic", metric="euclidean")
    clusterer.fit(each_dataset)
    #shuffle_corrected_labels.append(clusterer.probabilities_)
    shuffle_corrected_labels.append(clusterer.labels_)"""


    """clusterer = MyHdbscan(min_points=min_cluster_size)
    clusterer.fit_hdbscan(each_dataset)
    shuffle_corrected_labels.append(clusterer.labels_)"""

    for i in range(3):
        shuffled_indices = np.random.permutation(number_of_elements_to_shuffle)
        shuffled_data = each_dataset.copy()  # Make a copy to preserve the original dataset
        shuffled_data[:number_of_elements_to_shuffle] = shuffled_data[shuffled_indices]

        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True, approx_min_span_tree=False,
                        gen_min_span_tree=True, algorithm="generic", metric="euclidean")
        clusterer.fit(shuffled_data)

        """clusterer = MyHdbscan(min_points=min_cluster_size)
        clusterer.fit_hdbscan(shuffled_data)"""

        semi_labels = np.concatenate([shuffled_indices, np.arange(number_of_elements_to_shuffle, len(each_dataset))])

        if not np.array_equal(shuffled_data, each_dataset[semi_labels]):
            raise Exception("Relabeled rows are not identical")

        labelz = clusterer.labels_[semi_labels]
        label_counts = defaultdict(int)
        for label in labelz:
            label_counts[label] += 1
        label_counts_dict = dict(label_counts)
        sorted_label_counts = dict(sorted(label_counts_dict.items()))
        label_count_ndarray = np.array([item[1] for item in sorted(label_counts.items())])

        label_dicts.append(label_count_ndarray)
        #shuffle_corrected_labels.append(clusterer.probabilities_[semi_labels])
        shuffle_corrected_labels.append(labelz)
        shuffle_corrected_probabilities.append(clusterer.probabilities_[semi_labels])

print("All labels are equal:", check_identical_arrays(shuffle_corrected_labels))
print("All probabilites are equal:", check_identical_arrays(shuffle_corrected_probabilities))
print("All label counts are equal", check_identical_arrays(label_dicts))
#np.array_equal(np.sort(first_array), np.sort(array))
