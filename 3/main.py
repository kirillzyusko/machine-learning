import scipy.io as sio
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import optimize as opt


def h0x(X, theta):
    return np.dot(X, theta)


def cost_l2(theta, X, y, lamb=0):
    predictions = h0x(X, theta)
    squared_errors = np.sum(np.square(predictions - y))
    regularization = np.sum(lamb * np.square(theta[1:]))
    return (squared_errors + regularization) / (2 * len(y))


def gradient_l2(theta, X, y, lamb):
    predictions = h0x(X, theta)
    gradient = np.dot(X.transpose(), (predictions - y))
    regularization = lamb * theta
    regularization[0] = 0  # because formula for 0 member is different
    return (gradient + regularization) / len(y)


def learning_curves_chart(X_train, y_train, X_val, y_val, lambda_):
    m = len(y_train)
    train_err = np.zeros(m)
    val_err = np.zeros(m)
    for i in range(1, m):
        theta = opt.fmin_cg(cost_l2, np.zeros(X_train.shape[1]), gradient_l2, (X_train[0:i + 1, :], y_train[0:i + 1], lambda_), disp=False)
        train_err[i] = cost_l2(theta, X_train[0:i + 1, :], y_train[0:i + 1])
        val_err[i] = cost_l2(theta, X_val, y_val)
    plt.plot(range(2, m + 1), train_err[1:], c="r", linewidth=2)
    plt.plot(range(2, m + 1), val_err[1:], c="b", linewidth=2)
    plt.xlabel("number of training examples", fontsize=14)
    plt.ylabel("error", fontsize=14)
    plt.legend(["training", "validation"], loc="best")
    plt.axis([2, m, 0, 100])
    plt.grid()
    plt.show()


def polynom(x, degree):
    X_poly = np.zeros(shape=(len(x), degree))
    for i in range(0, degree):
        X_poly[:, i] = x.squeeze() ** (i + 1);
    return X_poly


if __name__ == '__main__':
    # 1
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'ex3data1.mat')
    dataset = sio.loadmat(file_path)
    x_train = dataset["X"]
    x_val = dataset["Xval"]
    x_test = dataset["Xtest"]

    # squeeze the target variables into one-dimensional arrays
    y_train = dataset["y"].squeeze()
    y_val = dataset["yval"].squeeze()
    y_test = dataset["ytest"].squeeze()

    # 2
    fig, ax = plt.subplots()
    ax.scatter(x_train, y_train)
    plt.xlabel("change in water level", fontsize=14)
    plt.ylabel("water flowing out of the dam", fontsize=14)
    plt.title("Training sample", fontsize=16)
    plt.show()

    # data preparation
    X_train = np.hstack((np.ones((len(x_train), 1)), x_train))
    theta = np.zeros(X_train.shape[1])

    # 3
    print(cost_l2(theta, X_train, y_train, 0))

    # 4
    print(gradient_l2(theta, X_train, y_train, 0))

    # 5
    theta = opt.fmin_cg(cost_l2, theta, gradient_l2, (X_train, y_train, 1000), disp=False)
    print(theta)

    h = np.dot(X_train, theta)
    fig, ax = plt.subplots()
    ax.scatter(x_train, y_train)
    ax.plot(X_train[:, 1], h, linewidth=2, color='red')
    plt.show()

    # 6
    X_val = np.hstack((np.ones((len(x_val), 1)), x_val))
    plt.title("Learning Curves for Linear Regression", fontsize=16)
    learning_curves_chart(X_train, y_train, X_val, y_val, 0)

    # 7
    x_train_poly = polynom(x_train, 8)
    x_val_poly = polynom(x_val, 8)
    x_test_poly = polynom(x_test, 8)

    # 8
    train_means = x_train_poly.mean(axis=0)
    train_std = np.std(x_train_poly, axis=0, ddof=1)

    x_train_poly = (x_train_poly - train_means) / train_std
    x_val_poly = (x_val_poly - train_means) / train_std
    x_test_poly = (x_test_poly - train_means) / train_std

    X_train_poly = np.hstack((np.ones((len(x_train_poly), 1)), x_train_poly))
    X_val_poly = np.hstack((np.ones((len(x_val_poly), 1)), x_val_poly))
    X_test_poly = np.hstack((np.ones((len(x_test_poly), 1)), x_test_poly))

    # 9
    theta = opt.fmin_cg(cost_l2, np.zeros(X_train_poly.shape[1]), gradient_l2, (X_train_poly, y_train, 0), disp=False)
    x = np.linspace(min(x_train) - 5, max(x_train) + 5, 1000)
    x_polynom = polynom(x, 8)
    print(x_polynom)
    x_polynom = (x_polynom - train_means) / train_std
    x_polynom = np.hstack((np.ones((len(x_polynom), 1)), x_polynom))

    # 10
    fig, ax = plt.subplots()
    plt.scatter(x_train, y_train, color='red')
    plt.plot(x, h0x(x_polynom, theta), linewidth=2)
    plt.xlabel("change in water level", fontsize=14)
    plt.ylabel("water flowing out ", fontsize=14)
    plt.title("Polynomial Fit", fontsize=16)
    plt.show()

    learning_curves_chart(X_train_poly, y_train, X_val_poly, y_val, 0)

    # 11
    # lambda = 1
    theta = opt.fmin_cg(cost_l2, np.zeros(X_train_poly.shape[1]), gradient_l2, (X_train_poly, y_train, 1), disp=False)
    x = np.linspace(min(x_train) - 5, max(x_train) + 5, 1000)
    x_polynom = polynom(x, 8)
    x_polynom = (x_polynom - train_means) / train_std
    x_polynom = np.hstack((np.ones((len(x_polynom), 1)), x_polynom))
    fig, ax = plt.subplots()
    plt.scatter(x_train, y_train, color='red')
    plt.plot(x, h0x(x_polynom, theta), linewidth=2)
    plt.xlabel("change in water level", fontsize=14)
    plt.ylabel("water flowing out ", fontsize=14)
    plt.title("Polynomial Fit (lambda=1)", fontsize=16)
    plt.show()

    learning_curves_chart(X_train_poly, y_train, X_val_poly, y_val, 1)

    # lambda = 100
    theta = opt.fmin_cg(cost_l2, np.zeros(X_train_poly.shape[1]), gradient_l2, (X_train_poly, y_train, 100), disp=False)
    x = np.linspace(min(x_train) - 5, max(x_train) + 5, 1000)
    x_polynom = polynom(x, 8)
    x_polynom = (x_polynom - train_means) / train_std
    x_polynom = np.hstack((np.ones((len(x_polynom), 1)), x_polynom))
    fig, ax = plt.subplots()
    plt.scatter(x_train, y_train, color='red')
    plt.plot(x, h0x(x_polynom, theta), linewidth=2)
    plt.xlabel("change in water level", fontsize=14)
    plt.ylabel("water flowing out ", fontsize=14)
    plt.title("Polynomial Fit (lambda=100)", fontsize=16)
    plt.show()

    learning_curves_chart(X_train_poly, y_train, X_val_poly, y_val, 100)

    # 12
    lambda_values = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    val_err = []
    for lamb in lambda_values:
        theta = opt.fmin_cg(cost_l2, np.zeros(X_train_poly.shape[1]), gradient_l2, (X_train_poly, y_train, lamb), disp=False)
        val_err.append(cost_l2(theta, X_val_poly, y_val))
    plt.plot(lambda_values, val_err, c="b", linewidth=2)
    plt.axis([0, len(lambda_values), 0, val_err[-1] + 1])
    plt.grid()
    plt.xlabel("lambda", fontsize=14)
    plt.ylabel("error", fontsize=14)
    plt.title("Validation Curve", fontsize=16)
    plt.show()

    # 13
    theta = opt.fmin_cg(cost_l2, np.zeros(X_train_poly.shape[1]), gradient_l2, (X_train_poly, y_train, 3), disp=False)
    test_error = cost_l2(theta, X_test_poly, y_test)
    print("Test Error: ", test_error, "| Regularized Polynomial (lambda=3))")
