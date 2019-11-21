from __future__ import division
from goto import with_goto
import scipy.io as sio
import scipy.misc as misc
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist


def generate_k_rand_centroids(k, n, min=0, max=8):
    centroids = []
    for i in range(k):
        centroid = []
        for j in range(n):
            centroid.append(random.randint(min, max))
        centroids.append(centroid)

    return np.array(centroids)


def generate_k_rand_centroids_from_dataset(X, K, min=0, max=8):
    m, n = X.shape[0], X.shape[1]
    centroids = np.zeros((K, n))

    for i in range(K):
        centroids[i] = X[np.random.randint(min, max),:]

    return centroids


def find_closest_centroid(X, centroids):
    K = centroids.shape[0]
    idx = np.zeros((X.shape[0], 1))
    temp = np.zeros((centroids.shape[0], 1))

    for i in range(X.shape[0]):
        for j in range(K):
            dist = X[i] - centroids[j]
            temp[j] = np.sum(dist**2)  # a^2 + b^2
        idx[i] = np.argmin(temp) + 1

    return idx


def compute_centroids(X, idx, K):
    m, n = X.shape[0], X.shape[1]
    centroids = np.zeros((K, n))
    count = np.zeros((K, 1))

    for i in range(m):
        index = int((idx[i]-1)[0])
        centroids[index] += X[i]
        count[index] += 1

    return centroids/count


def plot_k_means(X, initial_centroids, K, num_iters):
    """
    plots the data points with colors assigned to each centroid
    """
    m, n = X.shape[0], X.shape[1]
    idx = find_closest_centroid(X, initial_centroids)

    fig, ax = plt.subplots(nrows=num_iters, ncols=1, figsize=(6, 36))
    history = k_means_with_history(X, idx, K, num_iters)
    for i in range(num_iters):
        [centroids, idx] = history[i]
        # Visualisation of data
        color = "rgb"
        for k in range(1, K+1):
            grp = (idx == k).reshape(m, 1)
            ax[i].scatter(X[grp[:, 0], 0], X[grp[:, 0], 1], c=color[k-1], s=15)
        # visualize the new centroids
        ax[i].scatter(centroids[:, 0], centroids[:, 1], s=120, marker="x", c="black", linewidth=3)
        title = "Iteration Number " + str(i)
        ax[i].set_title(title)

    plt.tight_layout()
    plt.show()


def k_means(X, idx, K, num_iters):
    for i in range(num_iters):
        # Compute the centroids mean
        centroids = compute_centroids(X, idx, K)

        # assign each training example to the nearest centroid
        idx = find_closest_centroid(X, centroids)

    return [centroids, idx]


def k_means_with_history(X, idx, K, num_iters):
    history = []
    for i in range(num_iters):
        [centroids, idx] = k_means(X, idx, K, 1)
        history.append([centroids, idx])
        history.append(k_means(X, idx, K, 1))

    return history


@with_goto
def main():
    goto .task
    # 1
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'ex6data1.mat')
    dataset = sio.loadmat(file_path)
    X = dataset["X"]

    # 2
    K = 3
    initial_centroids = generate_k_rand_centroids(K, 2)
    print(initial_centroids)
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])  # use mock

    # 3
    print(find_closest_centroid(np.array([[3.38156267, 3.38911268]]), initial_centroids))
    centroids = find_closest_centroid(X, initial_centroids)

    # 4
    compute_centroids(X, centroids, K)

    # 5
    k_means(X, centroids, K, 10)

    # 6
    initial_centroids = generate_k_rand_centroids_from_dataset(X, K)
    plot_k_means(X, initial_centroids, K, 10)

    # 7
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'bird_small.mat')
    dataset = sio.loadmat(file_path)
    A = dataset["A"]
    # preprocess and reshape the image
    X = (A/255).reshape(128*128, 3)

    # 8
    K = 16
    num_iters = 10
    initial_centroids = generate_k_rand_centroids_from_dataset(X, K, 0, 16384)
    idx = find_closest_centroid(X, initial_centroids)
    [centroids, idx] = k_means(X, idx, K, num_iters)

    # 9
    X_recovered = X.copy()
    for i in range(1, K+1):
        X_recovered[(idx == i).ravel(), :] = centroids[i-1]
    # Reshape the recovered image into proper dimensions
    X_recovered = X_recovered.reshape(128, 128, 3)
    # Display the image
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(X.reshape(128, 128, 3))
    ax[1].imshow(X_recovered)
    plt.show()

    # 10
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'bird.png')
    matrix = misc.imread(file_path)
    X = (matrix/255).reshape(443*590, 3)
    K = 16
    num_iters = 10
    initial_centroids = generate_k_rand_centroids_from_dataset(X, K, 0, 16384)
    idx = find_closest_centroid(X, initial_centroids)
    [centroids, idx] = k_means(X, idx, K, num_iters)
    X_recovered = X.copy()
    for i in range(1, K+1):
        X_recovered[(idx == i).ravel(), :] = centroids[i-1]
    # Reshape the recovered image into proper dimensions
    X_recovered = X_recovered.reshape(443, 590, 3)
    # Display the image
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(X.reshape(443, 590, 3))
    ax[1].imshow(X_recovered)
    # np.savetxt('output.txt', idx)  # for next assignment
    # misc.imsave('output.jpg', X_recovered)
    plt.show()

    # 11
    label .task
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'bird-small.png')
    img = misc.imread(file_path)
    plt.imshow(img)
    plt.show()
    img = img / 255  # feature scaling

    points = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    distance_mat = pdist(points)

    Z = hierarchy.linkage(distance_mat, 'single')
    max_d = .3
    while max_d > 0.005:
        max_d *= .5
        print(max_d)
        clusters = fcluster(Z, max_d, criterion='distance')
        meshx, meshy = np.meshgrid(np.arange(128), np.arange(128))
        plt.axis('equal')
        plt.axis('off')
        plt.scatter(meshx, -(meshy - 128), c=clusters.reshape(128, 128), cmap='inferno', marker=',')
        plt.show()


if __name__ == '__main__':
    main()
