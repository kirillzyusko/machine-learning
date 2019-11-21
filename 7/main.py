from mpl_toolkits.mplot3d import Axes3D
from goto import with_goto
import scipy.io as sio
import matplotlib.image as img
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import numpy as np
from numpy.linalg import svd


def feature_normalize(X):
    means = np.mean(X, axis=0)
    X_norm = X - means
    stds = np.std(X_norm, axis=0)
    X_norm = X_norm / stds

    return means, stds, X_norm


def compute_covariance_matrix(X):
    return X.T.dot(X) / X.shape[0]


def pca(X):
    covariance_matrix = compute_covariance_matrix(X)
    U, S, V = svd(covariance_matrix, full_matrices=True, compute_uv=True)

    return U, S


def project_data(X, U, K):
    return X.dot(U[:, :K])


def recover_data(Z, U, K):
    return Z.dot(U[:, :K].T)


def grid_plot(X, dim):
    fig = plt.figure(figsize=(6, 6))
    M, N = X.shape

    gs = gridspec.GridSpec(dim, dim)
    gs.update(bottom=0.01, top=0.99, left=0.01, right=0.99,
              hspace=0.05, wspace=0.05)

    k = 0
    for i in range(dim):
        for j in range(dim):
            ax = plt.subplot(gs[i, j])
            ax.axis('off')
            ax.imshow(-X[k].reshape(int(np.sqrt(N)), int(np.sqrt(N))).T,
                      cmap=plt.get_cmap('Greys'),  # vmin=-1, vmax=1,
                      interpolation='nearest')  # ,alpha = 1.0)
            k += 1

    plt.show()


@with_goto
def main():
    goto .task
    label .task
    # 1
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'ex7data1.mat')
    dataset = sio.loadmat(file_path)
    X = dataset["X"]

    # 2
    # X[:, 0] - first column
    plt.scatter(X[:, 0], X[:, 1], marker="o")
    plt.show()

    # 3
    covariance_matrix = compute_covariance_matrix(X)
    print(covariance_matrix)

    # 4
    # Feature normalize
    # mu, sigma
    means, stds, X_norm = feature_normalize(X)
    # Run SVD
    U, S = pca(X_norm)
    print(U, S)

    # 5
    #  Draw the eigenvectors centered at mean of data. These lines show the
    #  directions of maximum variations in the dataset.
    fig, ax = plt.subplots()
    ax.plot(X[:, 0], X[:, 1], 'o', mew=0.25)

    for i in range(len(S)):
        ax.arrow(means[0], means[1], 1.5 * S[i]*U[0, i], 1.5 * S[i]*U[1, i],
                 head_width=0.25, head_length=0.2, fc='k', ec='k', lw=2, zorder=1000)

    ax.axis([0.5, 6.5, 2, 8])
    ax.set_aspect('equal')
    ax.grid(False)

    plt.show()

    print('Top principal component: U[:, 0] = [{:.6f} {:.6f}]'.format(U[0, 0], U[1, 0]))
    print(' (you should expect to see [-0.707107 -0.707107])')

    # 6
    # Project the data onto K = 1 dimension
    K = 1
    Z = project_data(X_norm, U, K)
    print('Projection of the first example: {:.6f}'.format(Z[0, 0]))
    print('(this value should be about    : 1.481274)')

    # 7
    X_rec = recover_data(Z, U, K)
    print('Approximation of the first example: [{:.6f} {:.6f}]'.format(X_rec[0, 0], X_rec[0, 1]))
    print('       (this value should be about  [-1.047419 -1.047419])')

    # 8
    plt.figure(figsize=(6, 6))
    plt.plot(X_norm.T[0], X_norm.T[1], 'bo', mfc='none', mec='b', ms=8, label='Original Data Points')
    plt.plot(X_rec.T[0], X_rec.T[1], 'ro', mfc='none', mec='r', ms=8, label='PCA Reduced Data Points')
    plt.xlabel('x1', fontsize=14)
    plt.ylabel('x2', fontsize=14)
    plt.legend(loc=4)
    for (x, y), (x_rec, y_rec) in zip(X_norm, X_rec):
        plt.plot([x, x_rec], [y, y_rec], 'k--', lw=1)
    plt.xlim(-4, 3)
    plt.ylim(-4, 3)
    plt.show()

    # 9
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'ex7faces.mat')
    dataset = sio.loadmat(file_path)
    X = dataset['X']

    # 10
    grid_plot(X, 10)

    # 11
    # mu, sigma
    means, stds, X_norm = feature_normalize(X)
    U, S = pca(X_norm)

    # 12, 13
    grid_plot(U.T, 6)

    # 14-15
    grid_plot(U.T, 10)

    # 16
    A = img.imread(os.path.join('data', 'output.jpg'))
    X = A.reshape(-1, 3)

    # 17
    # Sample 1000 random indexes (since working with all the data is
    #  too expensive. If you have a fast computer, you may increase this.
    sel = np.random.choice(X.shape[0], size=1000)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    idx = np.loadtxt(os.path.join('data', 'output.txt'))
    ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], cmap='rainbow', c=idx[sel], s=10)
    ax.set_title('Pixel dataset plotted in 3D.\nColor shows centroid memberships')

    plt.show()

    # 18
    mu, sigma, X_norm = feature_normalize(X)

    # PCA and project the data to 2D
    U, S = pca(X_norm)
    Z = project_data(X_norm, U, 2)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    ax.scatter(Z[sel, 0], Z[sel, 1], cmap='rainbow', c=idx[sel], s=32)
    ax.set_title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
    ax.grid(False)

    plt.show()


if __name__ == '__main__':
    main()
