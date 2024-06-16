import numpy as np
import matplotlib.pyplot as plt


def generate_n_simplex(n, centroid, radius):
    """# Lambda function to generate initial n-simplex vertices
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

    return simplex_vertices"""
    # (n+1) x n matrix, initial simplex vertices in n-dimensional space
    vertices = np.eye(n + 1)[:-1]

    # Normalize the vertices to be equidistant from the origin
    norm_factor = np.linalg.norm(vertices[0] - np.mean(vertices, axis=0))
    vertices = vertices / norm_factor

    # Calculate the distance between vertices in a unit simplex
    unit_distance = np.linalg.norm(vertices[0] - vertices[1])

    # Scale the vertices to have the desired radius distance between them
    scale_factor = radius / unit_distance
    vertices *= scale_factor

    # Move vertices so that the centroid is the given datapoint
    centroid = np.array(centroid)
    if centroid.shape[0] != vertices.shape[1]:
        raise ValueError("Dimension of centroid must match the number of simplex dimensions (n).")
    centroid_shift = centroid - np.mean(vertices, axis=0)
    vertices += centroid_shift

    return vertices


# Example usage:
n = 2  # Dimension of the simplex (n-simplex in n+1 dimensions)
starting_point = np.array([-5, -5, -5])  # Example starting point
simplex_vertices = generate_n_simplex(n, starting_point, 1)
print("Vertices of the {}-simplex with starting point:".format(n))
print(simplex_vertices)

simplex_vertices = [[-1, 1, 0], [0, 0, np.sqrt(3) / 2]]

plt.scatter(simplex_vertices[0], simplex_vertices[1])
plt.show()
