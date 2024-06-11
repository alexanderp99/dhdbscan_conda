"""
1. The mutual reachabilty distance calculation of HDBSCAN* is not symmetric
Background:
-stage1 constructs a new matrix where each element is the greater value between the core distance of a point (broadcast across the row) and the original pairwise distance from the distance_matrix. This means for each row, the core distance of the corresponding point is compared with each column entry (pairwise distance).
-The intuition behind result is to take the maximum between the transposed version of stage1 and the core distances transposed. This aims to ensure that the mutual reachability is the maximum of either the distance from i to j or from j to i. However, since core_distances may differ for different points and the broadcasting applies differently when transposed (i.e., across columns instead of rows), the final result result can end up being asymmetric.
The asymmetry stems from how core_distances (being a 1D array) is involved in comparisons differently in the original and the transposed matrices. When you transpose and compare with core_distances.T, the comparison is no longer between the same pairs in the opposite order, leading to potential mismatches.
"""



"""
2. The mst thus was not deterministic

One could that assume that if data is shuffled, the weight list of the ordered MST is identical to other msts. This was only the case after altering the mutual reachability distance

"""


"""
3. The building of the linkage matrix was not deterministic

If the linkage matrix is shuffled only for points with identical weights, the result is still not the same
"""


