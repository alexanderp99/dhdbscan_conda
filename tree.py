import numpy as np

class Node:
    def __init__(self, value, distance, size=1):
        self.value = value
        self.children = []
        self.size = size
        self.distance = distance

    def add_child(self, child):
        self.children.append(child)

class Tree:
    def __init__(self):
        self.root = None

    def build_from_linkage_matrix(self, L: np.ndarray):
        N = L.shape[0] + 2
        U = UnionFind(N)
        nodes = {i: Node(i, np.infty) for i in range(N)}

        last_node_idx = None

        for index in range(L.shape[0]):
            a = int(L[index, 0])
            b = int(L[index, 1])
            delta = L[index, 2]

            aa = U.find(a)
            bb = U.find(b)

            new_node = Node(-1, distance=delta, size=U.size[aa] + U.size[bb])  # Create a new internal node
            new_node.add_child(nodes[aa])
            new_node.add_child(nodes[bb])

            nodes[aa] = new_node
            nodes[bb] = new_node

            U.union(aa, bb)
            # After the union, the new_node represents the union of aa and bb
            nodes[U.find(aa)] = new_node
            last_node_idx = U.find(aa)
        # Find the root node
        self.root = nodes[last_node_idx]

    def print_tree(self, node=None, level=0):
        if node is None:
            node = self.root
        print(' ' * level * 2, f'Node {node.value} (Size: {node.size})')
        for child in node.children:
            self.print_tree(child, level + 1)

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
        self.size = [1] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)

        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
                self.size[root_u] += self.size[root_v]
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
                self.size[root_v] += self.size[root_u]
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1
                self.size[root_u] += self.size[root_v]

# Example usage:
data = np.array([[1, 2, 0.1], [4, 3, 0.2], [2, 3, 0.25], [4, 5, 0.3], [5, 6, 0.4]])
tree = Tree()
tree.build_from_linkage_matrix(data)
tree.print_tree()
