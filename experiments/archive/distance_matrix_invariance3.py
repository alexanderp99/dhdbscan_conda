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
datapoints = np.array([
    [ 1.          ,  1.          ,  1.         , 19.5        , 20.5        , 20.01874122, 19.0197266 , 32.90231755, 42.5470328 , 41.11265012, 37.88469348, 36.26637561],
    [ 1.          ,  1.          ,  1.         , 18.5        , 19.5        , 19.0197266 , 18.5        , 32.03221035, 42.89813516, 41.35516896, 38.27858409, 36.54107278],
    [ 1.          ,  1.          ,  1.         , 19.0197266  , 20.01874122 , 19.5       , 19.5        , 32.90175243, 43.53196564, 42.07174862, 38.87636529, 37.23401373],
    [19.5        , 18.5        , 19.0197266  ,  1.          ,  1.          ,  1.        , 18.5        , 18.5, 52.49761899, 49.40647731, 48.7954916 , 45.45327271],
    [20.5        , 19.5        , 20.01874122 ,  1.          ,  1.          ,  1.        , 19.0197266 , 18.5, 53.15072906, 50.        , 49.49747468, 46.09772229],
    [20.01874122 , 19.0197266  , 19.5        ,  1.          ,  1.          ,  1.        , 19.5        , 19.00066612, 53.48160462, 50.40121062, 49.76566867, 46.43944205],
    [19.0197266  , 18.5        , 19.5        , 18.5        , 19.0197266  , 19.5       , 18.5        , 18.50001246, 34.46059202, 31.06497066, 31.18889389, 27.39063895],
    [32.90231755 , 32.03221035 , 32.90175243 , 18.48140958 , 18.48140958 , 19.00066612, 18.50001246, 18.48140958, 49.4627385 , 45.15597967, 47.2394168 , 42.70904471],
    [42.5470328  , 42.89813516 , 43.53196564 , 52.49761899 , 53.15072906 , 53.48160462, 34.46059202, 49.4627385 ,  5.         ,  5.         ,  5.         ,  7.07106781],
    [41.11265012 , 41.35516896 , 42.07174862 , 49.40647731 , 50.         , 50.40121062, 31.06497066, 45.15597967,  5.         ,  5.         ,  7.07106781,  5.        ],
    [37.88469348 , 38.27858409 , 38.87636529 , 48.7954916  , 49.49747468 , 49.76566867, 31.18889389, 47.2394168 ,  5.         ,  7.07106781,  5.         ,  5.        ],
    [36.26637561 , 36.54107278 , 37.23401373 , 45.45327271 , 46.09772229 , 46.43944205, 27.39063895, 42.70904471,  7.07106781,  5.         ,  5.         ,  5.        ]
])


datasets = {
    'dataset1': {'dataset': datapoints, 'min_cluster_size': 2}
}
label_dicts = []

shuffle_corrected_labels = []
shuffle_corrected_probabilities = []

for dataset_name, each_datadict in datasets.items():
    each_dataset = each_datadict['dataset']
    min_cluster_size = each_datadict['min_cluster_size']

    for i in range(15):
        #shuffled_indices = np.random.permutation(len(each_dataset))
        shuffled_indices = np.arange(len(each_dataset))
        shuffled_data = each_dataset.copy()  # Make a copy to preserve the original dataset
        shuffled_data = shuffled_data[shuffled_indices]

        """clusterer = HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True, approx_min_span_tree=False,
                        gen_min_span_tree=True, algorithm="generic", metric="euclidean")"""
        clusterer = MyHdbscan(min_points=min_cluster_size)
        #clusterer.fit_hdbscan_distance_matrix(shuffled_data)
        #clusterer.fit_hdbscan(shuffled_data)
        clusterer.fit_hdbscan_mutual_reachability(datapoints)
        #clusterer.fit(shuffled_data)



        """clusterer.condensed_tree_.plot(select_clusters=True)
        plt.title("Hdbscan condensed tree")
        plt.show()"""

        labels = clusterer.labels_[shuffled_indices]
        #probabilities = clusterer.probabilities_
        probabilities = clusterer.probabilities

        """num_colors = len(set(labels))
        palette = plt.cm.viridis(np.linspace(0, 1, num_colors))
        cluster_colors = [palette[col] if col >= 0 else (0.5, 0.5, 0.5) for col in labels]
        plt.scatter(each_dataset.T[0], each_dataset.T[1], c=cluster_colors)
        plt.title("Hdbscan scatterplot matplotlib")
        plt.show()"""

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
