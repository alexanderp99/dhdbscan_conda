import numpy as np
from itertools import groupby, tee


class Node:
    def __init__(self, value, distance, size=1):
        self.value = value
        self.children = []
        self.size = size
        self.distance = distance
        self.lambd = 1 / self.distance
        self.label = None
        self.delta = 0
        self.parent = None
        self.stability = None

    def add_child(self, child):
        child.parent = self
        self.children.append(child)


class Cluster:
    def __init__(self):
        self.children = []
        self.label = None
        self.parent = None
        self._delta_creation = 0
        self._delta_end = 0
        self.values = []
        self._stability = 0
        self.is_selected = False

    @property
    def delta_creation(self):
        return self._delta_creation

    @delta_creation.setter
    def delta_creation(self, value):
        self._delta_creation = value
        self._update_stability()

    @property
    def delta_end(self):
        return self._delta_end

    @delta_end.setter
    def delta_end(self, value):
        self._delta_end = value
        self._update_stability()

    @property
    def stability(self):
        return self._stability

    @stability.setter
    def stability(self, value):
        self._stability = value

    def _update_stability(self):
        self._stability = abs(self._delta_end - self._delta_creation)

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def __str__(self):
        return f"Cluster: {self.values}"

    def __repr__(self):
        return f"Cluster: {self.values}"


class Tree:
    def __init__(self):
        self.root = None
        self.minimum_cluster_size = 2

    def build_from_linkage_matrix(self, L: np.ndarray):
        N = L.shape[0]
        U = UnionFind(N)
        nodes = {i: Node(i, np.infty) for i in range(L.shape[0])}

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

    def build_from_linkage_matrix2(self, L):
        N = L.shape[0] + 2  # +2
        U = UnionFind(N)

        # num_datapoints = max(max(L[:,:1]),max(L[:,1:2])) +1 # #+1 because points start at 0
        nodes = {i: Node(i, np.infty) for i in range(N)}
        # nodes = {index: Node(index, np.inf, L[index, 0], L[index, 1]) for index in range(len(L))}
        last_node_idx = None
        index = 0

        for delta, group in groupby(L, key=lambda x: x[2]):

            group1, group2 = tee(group)

            group2 = list(group2)
            # Calculate the number of elements in the group by exhausting group1
            group_size = sum(1 for _ in group1)
            if group_size > 1:

                distance = float(group2[0][2:])# shows distance for all datapoints
                U2 = UnionFind(N)
                di = {}
                for each_row in group2:

                    first_value = U.find(int(each_row[0]))
                    second_value = U.find(int(each_row[1]))

                    #if U2.find(first_value) != first_value or U2.find(second_value) != second_value:


                    U2.union(first_value, second_value)
                    pos = U2.find(first_value)
                    if pos not in di:
                        di[pos] = []
                        di[pos].append(each_row[:2])
                    else:
                        di[pos].append(each_row[:2])



                for each_ntree in di.values():

                    unique_elements, unique_counts = np.unique(np.array(each_ntree).flatten(),return_counts=True)

                    # next check
                    duplicate_element = int(unique_elements[unique_counts > 1][0])# choose first duplicate

                    unique_elements = np.unique(np.array(each_ntree).flatten()).astype(int)
                    union_find_results = [U.find(each_element) for each_element in unique_elements]
                    union_find_sizes = [U.size[each_element] for each_element in union_find_results]


                    new_node = Node(-1, distance=distance, size=sum(union_find_sizes))  # Create a new internal node

                    for each_union_find_result in union_find_results:
                        new_node.add_child(nodes[each_union_find_result])
                        nodes[each_union_find_result] = new_node
                        U.union(duplicate_element, each_union_find_result)
                    # After the union, the new_node represents the union of aa and bb
                    last_node_idx = U.find(duplicate_element)

                    index += len(each_ntree)

                """
                index_items = np.array(group2)[:, :2].flatten()
                items_have_shared_components = (len(np.unique(index_items)) != len(index_items))
                unique_elements, unique_counts = np.unique(index_items, return_counts=True)

                # next check
                duplicate_element = unique_elements[unique_counts > 1]
                non_duplicate_elements = unique_elements[unique_counts == 1]
                if len(duplicate_element) > 1:
                    print("exeption. Can not handle multiple duplicates")
                # duplicate_element = int(duplicate_element)

                union_find_values = []
                all_non_duplicates_not_in_same_cluster = True
                for each_value in unique_elements:
                    union_find_value = U.find(int(each_value))
                    u_size = U.size[union_find_value]
                    if u_size == 1:
                        all_non_duplicates_not_in_same_cluster = False
                    else:
                        union_find_values.append(each_value)"""

                if not items_have_shared_components and all_non_duplicates_not_in_same_cluster:
                    for item in group2:
                        a = int(item[0])
                        b = int(item[1])
                        delta = item[2]

                        aa = U.find(a)
                        bb = U.find(b)

                        new_node = Node(-1, distance=delta, size=U.size[aa] + U.size[bb])  # Create a new internal node
                        new_node.add_child(nodes[aa])
                        new_node.add_child(nodes[bb])

                        nodes[aa] = new_node
                        nodes[bb] = new_node

                        U.union(aa, bb)
                        # After the union, the new_node represents the union of aa and bb
                        last_node_idx = U.find(aa)

                        index += 1
                else:
                    # print("attention, we have a conflict")

                    if len(union_find_values) >= 1:
                        # items have not shared components (thus all 4 unique), but they all belong to the same cluster
                        pass

                    first_merge_num_childs = None  # This guarantees, that other connected components that connect to the same component on the same level, have the same number of children (split condition)

                    # I am ussuming the third unique point is in relation with the douplicate point.

                    a = int(unique_elements[0])
                    b = int(unique_elements[1])
                    c = int(unique_elements[2])
                    delta = group2[0][2]

                    aa = U.find(a)
                    bb = U.find(b)
                    cc = U.find(c)

                    """if first_merge_num_childs is None:
                        first_merge_num_childs = U.size[aa] + U.size[bb]
                    result_arr[index, 3] = first_merge_num_childs
                    U.union(aa, bb)

                    index += 1
                    """

                    # new_node = Node(-1, distance=delta, size=U.size[aa] + U.size[bb] + 1)  # +1 for C Node
                    new_node = Node(-1, distance=delta,
                                    size=U.size[aa] + U.size[bb] + U.size[cc])  # Root node was all children

                    new_node.add_child(nodes[aa])
                    new_node.add_child(nodes[bb])
                    new_node.add_child(nodes[cc])

                    nodes[aa] = new_node
                    nodes[bb] = new_node
                    nodes[cc] = new_node

                    U.union(aa, bb)
                    U.union(aa, cc)

                    # After the union, the new_node represents the union of aa and bb
                    last_node_idx = U.find(aa)

                    index += 2  # 2 iterations simulated

            else:
                for item in group2:
                    a = int(item[0])
                    b = int(item[1])
                    delta = item[2]

                    aa = U.find(a)
                    bb = U.find(b)

                    new_node = Node(-1, distance=delta, size=U.size[aa] + U.size[bb])  # Create a new internal node
                    new_node.add_child(nodes[aa])
                    new_node.add_child(nodes[bb])

                    nodes[aa] = new_node
                    nodes[bb] = new_node

                    U.union(aa, bb)
                    # After the union, the new_node represents the union of aa and bb
                    last_node_idx = U.find(aa)

                    index += 1

        root = nodes[last_node_idx]

        return root

    def print_tree(self, node=None, level=0):
        if node is None:
            node = self.root
        print(' ' * level * 2,
              f'Node {node.value} (Size: {node.size}) (Dist:{node.distance}) (Label:{node.label}) (Delta:{node.delta}) (Lambda:{node.lambd})')
        for child in node.children:
            self.print_tree(child, level + 1)

    def print_cluster_tree(self, node=None, level=0):
        print(' ' * level * 2,
              f'Cluster Label:{node.label} (Values: {node.values}) (Delta creat: {node.delta_creation}) (Delta End:{node.delta_end}) (Stability: {node.stability}) (IsCluster: {node.is_selected}) (Parent: {node.parent.label if node.parent else node.parent})')
        for child in node.children:
            self.print_cluster_tree(child, level + 1)

    def condense_tree(self, root_node, label=1):
        # Spurious if fewer than min cluster size objects
        if label == 1:
            root_node.label = label
        non_spurious_clusters = []
        for child_node in root_node.children:
            if child_node.size < self.minimum_cluster_size:
                child_node.label = None
            elif child_node.size >= self.minimum_cluster_size:
                child_node.label = label  # assume cluster shrunk
                non_spurious_clusters.append(child_node)

        if len(non_spurious_clusters) >= 2:
            for child_node in root_node.children:
                if child_node.size >= self.minimum_cluster_size:
                    label += 1
                    child_node.label = label  # true cluster split means new label
                    label = self.condense_tree(child_node, label)
        else:
            for child_node in root_node.children:
                if child_node.size >= self.minimum_cluster_size:
                    child_node.label = label  # same label - assume cluster shrunk
                    self.condense_tree(child_node, label)
        return label

    def extract_stable_clusters(self, root_node):

        # Setting the cluster leaf delta
        def dfs(node):

            if all(child.value >= 0 for child in node.children) and len(node.children) == 2:
                node.stability = 1
            else:
                for each_child_node in node.children:
                    dfs(each_child_node)

        dfs(root_node)

    def calculate_clusters(self, node):

        root_cluster = Cluster()
        root_cluster.delta_creation = node.lambd
        root_cluster.label = node.label

        def traverse(node, current_cluster):
            for child in node.children:
                if child.value == -1 and child.label == node.label:
                    current_cluster.delta_end = child.lambd

                    # child_node_values = [each_child.value for each_child in child.children if each_child.value >= 0]
                    # current_cluster.values.append(child_node_values)

                    traverse(child, current_cluster)
                elif child.value == -1 and child.label != node.label:
                    child_cluster = Cluster()
                    child_cluster.delta_creation = child.lambd
                    child_cluster.label = child.label
                    child_cluster.parent = current_cluster
                    # child_node_values = [each_child.value for each_child in child.children if each_child.value >= 0]
                    # current_cluster.values.append(child_node_values)

                    current_cluster.add_child(child_cluster)
                    traverse(child, child_cluster)
                elif child.value >= 0:
                    current_cluster.values.append(child.value)

        traverse(node, root_cluster)
        return root_cluster

    def calculate_stabilities(self, root_cluster):
        def dfs(node):
            if len(node.children) == 0:
                node.is_selected = True
            else:
                for each_child_node in node.children:
                    dfs(each_child_node)
                stabilities = []
                if node.children[0].is_selected:  #
                    for each_child_node in node.children:
                        stabilities.append(each_child_node.stability)
                if np.sum(stabilities) > node.stability:
                    node.stability = np.sum(stabilities)  # Parent stabilites set to child stabilities
                else:
                    node.is_selected = True
                    for each_child_node in node.children:
                        each_child_node.is_selected = False

        dfs(root_cluster)
        return

    def extractClusters(self, cluster_root):

        cluster_list = []

        def dfs(node):

            for each_child_node in node.children:
                if each_child_node.is_selected:
                    cluster_list.append(each_child_node)
                dfs(each_child_node)

        dfs(cluster_root)

        return cluster_list


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


def run_dhdbscan_from_mst(mst):
    do_printing = True

    tree = Tree()
    node = tree.build_from_linkage_matrix2(mst)
    if do_printing:
        tree.print_tree(node)
        print("\n\n\n")
    tree.condense_tree(node)
    # tree.extract_stable_clusters(node)
    if do_printing:
        tree.print_tree(node)
        print("\n\n\n")
    cluster = tree.calculate_clusters(node)
    if do_printing:
        tree.print_cluster_tree(cluster)
        print("\n\n\n")
    tree.calculate_stabilities(cluster)
    if do_printing:
        tree.print_cluster_tree(cluster)
    cluster_list = tree.extractClusters(cluster)
    if do_printing:
        print(cluster_list)
    return cluster_list
