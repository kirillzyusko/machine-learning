from __future__ import division
from goto import with_goto
from scipy.stats import multivariate_normal
import scipy.stats as stats
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import numpy as np


def estimate_gaussian(X):
    # Useful variables
    m, n = X.shape
    mu = np.mean(X, axis=0)
    # For Sigma2,np.sum requires an axis else it flattens the array and takes the sum which is wrong.
    sigma2 = (1/m)*(np.sum((X-mu)**2, axis=0))

    return mu, sigma2


def select_threshold(y_val, p_val):
    best_epsilon, best_F1 = 0, 0

    step_size = (max(p_val) - min(p_val)) / 1000
    for epsilon in np.arange(p_val.min(), p_val.max(), step_size):
        predictions = (p_val < epsilon)[:, np.newaxis]
        tp = np.sum(predictions[y_val == 1] == 1)
        fp = np.sum(predictions[y_val == 0] == 1)
        fn = np.sum(predictions[y_val == 1] == 0)

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)

        F1 = 2 * prec * rec / (prec + rec)

        if F1 > best_F1:
            best_epsilon = epsilon
            best_F1 = F1

    return best_epsilon, best_F1


@with_goto
def main():
    goto .task
    label .task
    # 1
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'ex8data1.mat')
    dataset = sio.loadmat(file_path)
    X = dataset["X"]
    Xval = dataset["Xval"]
    yval = dataset["yval"]

    # 2
    plt.scatter(X[:, 0], X[:, 1], marker="o", s=16)
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Throughput (mb/s)")
    plt.show()

    # 3
    mu, sigma2 = estimate_gaussian(X)

    # 4
    p = multivariate_normal(mu, np.diag(sigma2))
    print(mu, sigma2)

    # 5
    xs, ys = np.mgrid[0:30:0.1, 0:30:0.1]
    pos = np.empty(xs.shape + (2,))
    pos[:, :, 0] = xs
    pos[:, :, 1] = ys

    plt.figure(figsize=(8, 6))
    plt.plot(X.T[0], X.T[1], 'o', ms=4)
    plt.contour(xs, ys, p.pdf(pos), 10.**np.arange(-21, -2, 3))
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.show()

    # 6
    p_val = p.pdf(Xval)
    epsilon, F1 = select_threshold(yval, p_val)
    print("Best epsilon found using cross-validation:", epsilon)
    print("Best F1 on Cross Validation Set:", F1)

    # 7
    outliers = X[p.pdf(X) < epsilon]
    plt.figure(figsize=(8, 6))
    plt.plot(X.T[0], X.T[1], 'o', ms=4)
    plt.plot(outliers.T[0], outliers.T[1], 'o', ms=18, mfc='none', mec='r')
    plt.contour(xs, ys, p.pdf(pos), 10.**np.arange(-21, -2, 3))
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.show()

    # 8
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'ex8data2.mat')
    dataset = sio.loadmat(file_path)
    X, Xval, yval = dataset['X'], dataset['Xval'], dataset['yval'][:, 0]

    # 9
    mu, sigma2 = estimate_gaussian(X)

    # 10
    p = multivariate_normal(mu, np.diag(sigma2))

    # 11
    p_val = p.pdf(Xval)
    epsilon, F1 = select_threshold(yval, p_val)
    print('Best epsilon found using cross-validation: %.2e' % epsilon)
    print('Best F1 on Cross Validation Set          : %f\n' % F1)
    print('  (you should see a value epsilon of about 1.38e-18)')
    print('  (you should see a Best F1 value of       0.615385)')

    # 12
    print('\n# Anomalies found: %d' % np.sum(p.pdf(X) < epsilon))


if __name__ == '__main__':
    main()
