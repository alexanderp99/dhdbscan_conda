import numpy as np


def simplex_coordinates1(m):
    """
    Computes the Cartesian coordinates of simplex vertices in m-dimensional space.

    Args:
    m (int): The spatial dimension of the simplex.

    Returns:
    np.ndarray: An m x (m+1) matrix where each column represents the coordinates of a simplex vertex.
    """
    x = np.zeros([m, m + 1])

    for k in range(m):
        # Set x[k, k] so that the sum of squares up to the k-th element is 1.
        s = 0.0
        for i in range(k):
            s += x[i, k] ** 2
        x[k, k] = np.sqrt(1.0 - s)

        # Set x[k, j] for j from k+1 to m using the fact that the dot product x_k * x_j = -1/m
        for j in range(k + 1, m + 1):
            s = 0.0
            for i in range(k):
                s += x[i, k] * x[i, j]
            x[k, j] = (-1.0 / float(m) - s) / x[k, k]

    return x


# Example usage
m = 2  # Dimensionality of the simplex
vertices = simplex_coordinates1(m)
print("Vertices of the simplex:")
print(vertices)

import matplotlib.pyplot as plt

plt.scatter(vertices[0], vertices[1])
plt.show()
