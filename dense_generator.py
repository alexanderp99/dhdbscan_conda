from DENSIRED import datagen
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as data
import random
from sklearn import datasets
from sklearn.manifold import TSNE


def generate_custom():
    return np.array([5, 4.7, 4.5, 4.4, 0, 0.1, 0.2, 0.3, -5, -4.8, -4.7, -4.6]).reshape(-1, 1), np.array([5.1, 5.2, 5.3, 5.45, 0.5, 0.4, 0.7, 0.2, -5.1, -5.2, - 5.3,- 5.4]).reshape(-1, 1)

def generate_custom_bug():
    "when we run this we get a bug. Reason maybe: identical distances"
    return np.array([5, 4.7, 4.5, 4.4, 0, 0.1, 0.2, 0.3, -5, -4.8, -4.7, -4.6]).reshape(-1, 1), np.array([5, 5, 5, 5, 0, 0, 0, 0, -5, -5, - 5,- 5]).reshape(-1, 1)


def generate_multidimensional():
    digits = datasets.load_digits()
    data = digits.data
    projection = TSNE().fit_transform(data)
    return projection


def generate_density_data():
    data = datagen.densityDataGen(clunum=8, core_num=15, ratio_noise=0.1, dim=2, seed=42).generate_data(300)
    datax = data[:, 0:1]
    datay = data[:, 1:2]
    # third column are the labels
    return datax.reshape(-1, 1), datay.reshape(-1, 1)


def generate_meshgrid(lower_bound=1, upper_bound=25, xdim=25, ydim=25):
    x, y = np.meshgrid(np.linspace(lower_bound, upper_bound, xdim), np.linspace(lower_bound, upper_bound, ydim))
    return x.reshape(-1, 1), y.reshape(-1, 1)  # make data rows instead of columns


def generate_two_moons():
    random_seed = 54
    np.random.seed(random_seed)
    random.seed(random_seed)
    moons, _ = data.make_moons(n_samples=40, noise=0.05)
    blobs, _ = data.make_blobs(n_samples=40, centers=[(-0.75, 2.25), (1.0, 2.0)], cluster_std=0.25)
    test_data = np.vstack([moons, blobs])
    # plt.scatter(test_data.T[0], test_data.T[1])
    # plt.show()
    return test_data.T[0].reshape(-1, 1), test_data.T[1].reshape(-1, 1)  # make data rows instead of columns


def plot_meshgrid(x, y):
    plt.figure()
    plt.scatter(x, y, s=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot of Points in 2D')
    plt.show()


if __name__ == "__main__":
    x, y = generate_meshgrid()
    plot_meshgrid(x, y)
    generate_two_moons()
