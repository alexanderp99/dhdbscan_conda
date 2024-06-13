from hdbscan import HDBSCAN
from dense_generator import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data
import seaborn as sns
from dhdbscan.DHDBSCAN import DHDBSCAN

sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()


def create_datasets(data_generators):
    datasets = {}
    for name, (generator, min_cluster_size) in data_generators.items():
        x, y = generator()
        dataset = np.hstack((x, y))  # Merging x and y columns
        datasets[name] = {'dataset': dataset, 'min_cluster_size': min_cluster_size}
    return datasets


data_generators = {
    "custom": (generate_custom,3),
    #"two_moons": (generate_two_moons, 5),
    #"grid": (generate_meshgrid, 5),
    #"density": (generate_density_data, 10)
}

datasets = create_datasets(data_generators)

# Define boolean variables to control plotting
plot_minimum_spanning_tree = True
plot_single_linkage_tree = False
plot_condensed_tree = False
plot_cluster_scatter_matplotlib = True
plot_cluster_scatter_sns = False
plot_dhdbscan_scatter = True

for dataset_name, each_datadict in datasets.items():
    each_dataset = each_datadict['dataset']
    min_cluster_size = each_datadict['min_cluster_size']

    dhdbscan_labels = DHDBSCAN(min_points=min_cluster_size).fit_dhdbscan(each_dataset)

    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True, approx_min_span_tree=False,
                        gen_min_span_tree=True, algorithm="generic", metric="euclidean")
    clusterer.fit(each_dataset)
    #In contrast, if you are getting lots of small clusters, but believe there should be some larger scale structure (or the possibility of no structure), consider the allow_single_cluster option

    if plot_minimum_spanning_tree:
        clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                              edge_alpha=0.6,
                                              node_size=80,
                                              edge_linewidth=2)
        plt.show()

    if plot_single_linkage_tree:
        clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
        plt.title("Hdbscan single linkage tree")
        plt.show()

    if plot_condensed_tree:
        clusterer.condensed_tree_.plot(select_clusters=True)
        plt.title("Hdbscan condensed tree")
        plt.show()

    labels = clusterer.labels_
    probabilities = clusterer.probabilities_

    if plot_dhdbscan_scatter:
        dhdbscan_labels
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

        num_colors = len(set(dhbscan_labels))
        palette = plt.cm.viridis(np.linspace(0, 1, num_colors))
        cluster_colors = [palette[col] if col >= 0 else (0.5, 0.5, 0.5) for col in dhbscan_labels]
        plt.scatter(each_dataset.T[0], each_dataset.T[1], c=cluster_colors)
        plt.title("DHdbscan scatterplot")
        plt.show()


    if plot_cluster_scatter_matplotlib:
        num_colors = len(set(labels))
        palette = plt.cm.viridis(np.linspace(0, 1, num_colors))
        cluster_colors = [palette[col] if col >= 0 else (0.5, 0.5, 0.5) for col in labels]
        plt.scatter(each_dataset.T[0], each_dataset.T[1], c=cluster_colors)
        plt.title("Hdbscan scatterplot matplotlib")
        plt.show()

    if plot_cluster_scatter_sns:
        palette = sns.color_palette()
        cluster_colors = [sns.desaturate(palette[col], sat)
                          if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
                          zip(labels, probabilities)]
        plt.scatter(each_dataset.T[0], each_dataset.T[1], c=cluster_colors)
        plt.title("Hdbscan scatterplot sns")
        plt.show()

