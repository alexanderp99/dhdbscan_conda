import numpy as np
import networkx as nx
from itertools import groupby
from collections import defaultdict

from itertools import groupby, tee

class UnionFind:
    """Source: https://yuminlee2.medium.com/union-find-algorithm-ffa9cd7d2dba"""
    def __init__(self, numOfElements):
        numOfElements += 5 # index error
        self.parent = self.makeSet(numOfElements)
        self.size = [1] * numOfElements
        self.count = numOfElements

    def makeSet(self, numOfElements):
        return [x for x in range(numOfElements)]

    # Time: O(logn) | Space: O(1)
    def find(self, node):
        while node != self.parent[node]:
            # path compression
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    # Time: O(1) | Space: O(1)
    def union(self, node1, node2):
        root1 = self.find(node1)
        root2 = self.find(node2)

        # already in the same set
        if root1 == root2:
            return

        if self.size[root1] > self.size[root2]:
            self.parent[root2] = root1
            self.size[root1] += 1
        else:
            self.parent[root1] = root2
            self.size[root2] += 1

        self.count -= 1


def label_(L):
    result_arr = np.zeros((L.shape[0], 4))
    N = L.shape[0] + 1
    U = UnionFind(N)

    index = 0
    for delta, group in groupby(L, key=lambda x: x[2]):

        group1, group2 = tee(group)

        group2 = list(group2)
        # Calculate the number of elements in the group by exhausting group1
        group_size = sum(1 for _ in group1)
        if group_size > 1:
            print("attention")
            expanded_group = None  # Start with an empty placeholder for the array

            for idx, row in enumerate(group2):
                a_size = U.size[U.find(int(row[0]))]
                b_size = U.size[U.find(int(row[1]))]

                # Create a new row that includes the additional size sum
                new_row = np.append(row, a_size + b_size)

                # Stack the new row onto the expanded_group array
                if expanded_group is None:
                    expanded_group = new_row.reshape(1, -1)  # Initialize the array with the first row shaped correctly
                else:
                    expanded_group = np.vstack((expanded_group, new_row.reshape(1, -1)))

            # Sort expanded_group by the third column, which is 'delta' originally
            sorted_size = list(expanded_group[np.argsort(expanded_group[:, 2])])
            group2 = sorted_size

        for item in group2:
            a = int(item[0])
            b = int(item[1])
            delta = item[2]

            aa = U.find(a)
            bb = U.find(b)

            result_arr[index, 0] = a
            result_arr[index, 1] = b
            result_arr[index, 2] = delta
            result_arr[index, 3] = U.size[aa] + U.size[bb]


            U.union(aa, bb)

            index += 1

    return result_arr


def process_mstext(mstext, n_vertices, min_points):
    print("hi")


# Example usage
mstext = np.array([[0, 1, 0.1], [3, 2, 0.2], [1, 2, 0.2], [3, 4, 0.3]])
#mstext = np.array([[0, 1, 0.1], [1, 2, 0.2], [3, 2, 0.2], [3, 4, 0.3]])
l = []
l.append(np.array([[0, 1, 0.1], [3, 2, 0.2], [1, 2, 0.2], [3, 4, 0.3]]))
l.append(np.array([[0, 1, 0.1], [1, 2, 0.2], [3, 2, 0.2], [3, 4, 0.3]]))
n_vertices = 5
min_points = 5
results = []
for each_l in l:
    res = label_(mstext)
    results.append(res)
print("hi")
