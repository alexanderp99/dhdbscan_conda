import numpy as np


dhdbscan_dict = {}
for idx, each_cluster in enumerate(dhdbscan_labels):
    dhdbscan_dict[idx] = each_cluster.values
dhdb_label_counts = np.sort([len(each_dict_value) for each_dict_value in dhdbscan_dict.values()])

value_to_cluster = {}
for i, cluster in enumerate(dhdbscan_labels):
    for value in cluster.values:
        value_to_cluster[value] = i

# Generate the list of labels
num_values = each_dataset.shape[0]
dhbscan_labels = [value_to_cluster.get(i, -1) for i in range(num_values)]