import numpy as np
from n_tree import *
from dhdbscan.DHDBSCAN import DHDBSCAN

# Example usage:
data = np.array(
    [[4, 7, 0.001], [7, 8, 0.002], [7, 9, 0.003], [7, 10, 0.004], [11, 12, 0.005], [11, 13, 0.005], [11, 14, 0.005],
     [11, 9, 0.006], [1, 2, 0.1], [4, 3, 0.2], [2, 3, 0.25], [4, 5, 0.3], [5, 6, 0.4]])
cluster_list = run_dhdbscan_from_mst(data)

dhdbscan_dict = {}
for idx, each_cluster in enumerate(cluster_list):
    dhdbscan_dict[idx] = each_cluster.values
dhdb_label_counts = np.sort([len(each_dict_value) for each_dict_value in dhdbscan_dict.values()])

dhdbscan = DHDBSCAN(min_points=2).fit_mst(data)
labels, all_label_counts = np.unique(dhdbscan.labels_, return_counts=True)
hdb_label_counts = all_label_counts[1:]  # without noise label
print(f"All labels are identical: {all(np.equal(dhdb_label_counts, hdb_label_counts))}")

"""
- korrekte label wie bei HDBSCAN

-Datasets aus dem HDBSCAN paper
-Handgemachte Beispiele (auf Grid, oder grid(100x100 punkte))
-Datengenerator

-Daten
-3 Ergebnisse (Verschiedene Algorithmen)
-'mein' Ergebnis ist das korrekte
"""