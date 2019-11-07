from __future__ import division
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import scipy.io as sio


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cost_function(theta, X, y):
    m = len(y)
    h_theta = sigmoid(np.dot(X, theta))
    # J = (1 / m) * ((-y' * log(h_theta)) - (1 - y)' * log(1 - h_theta));
    J = (1 / m) * ((np.dot(-y.T, np.log(h_theta))) - np.dot((1 - y).T, np.log(1 - h_theta)))
    return J


def cost_function_regularized(theta, X, y, lambda_=0):
    m = len(y)
    h_theta = sigmoid(np.dot(X, theta))
    J = (1 / m) * ((np.dot(-y.T, np.log(h_theta))) - np.dot((1 - y).T, np.log(1 - h_theta))) + (lambda_ / (2 * m)) * np.sum(theta[1:]**2)
    return J


def h0x(X, theta):
    return sigmoid(np.dot(X.T, theta))


def polynom_multi_var(p1, p2):
    def multiply(x):  # 6 combination
        return (x[0] ** p1) * (x[1] ** p2)

    return ['(x1^%s)*(x2^%s)' % (p1, p2), multiply]


def gradient(theta, X, y):
    # grad = (1 / m) * (h_theta - y)' * X;
    m = len(y)
    h_theta = sigmoid(np.dot(X, theta))
    return (1 / m) * np.dot((h_theta - y).T, X)


def gradient_regularized(theta, X, y, lambda_=0):
    m = len(y)
    grad = np.zeros([m, 1])
    grad = (1 / m) * np.dot(X.T, (sigmoid(np.dot(X, theta)) - y))
    grad[1:] = grad[1:] + (lambda_ / m) * theta[1:]
    return grad


def predict_number(X, theta):
    return np.argmax(np.dot(X, theta.T))


if __name__ == '__main__':
    # 1
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'ex2data1.txt')
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :-1]  # first 2 column
    y = data.iloc[:, 2]  # last column
    data.head()

    # 2
    admitted = y == 1
    failed = y != 1
    adm = plt.scatter(X[admitted][0].values, X[admitted][1].values)
    not_adm = plt.scatter(X[failed][0].values, X[failed][1].values)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))
    plt.show()

    # 3
    (m, n) = X.shape
    X = np.hstack((np.ones((m, 1)), X))
    y = y[:, np.newaxis]
    theta = np.zeros((n + 1, 1))  # [[0.] [0.] [0.]]
    print('Cost at initial theta (zeros): ', cost_function(theta, X, y)[0][0])
    # test (from an Octave)
    print('Expected gradients (approx): [-0.1000, -12.0092, -11.2628]')
    print('Real gradient: %s' % gradient(theta, X, y))
    # Compute and display cost and gradient with non-zero theta
    test_theta = np.array([[-24], [.2], [.2]])
    print('Expected cost (approx): 0.218')
    print('Cost at test theta: %s' % cost_function(test_theta, X, y)[0][0])

    # 4
    temp = optimize.fmin_tnc(
        func=cost_function,
        x0=theta.flatten(),
        fprime=gradient,
        args=(X, y.flatten())
    )
    # the output of above function is a tuple whose first element contains the optimized values of theta
    theta_optimized = temp[0]
    print(theta_optimized)

    temp = optimize.minimize(cost_function, theta.flatten(), (X, y.flatten()), method='Nelder-Mead')
    print(temp.x)

    # Brovden Fletcher Goldfarb Shanno alghoritm
    theta_optimized = optimize.fmin_bfgs(
        cost_function,
        theta.flatten(),
        gradient,
        (X, y.flatten())
    )
    print(theta_optimized)

    # 5
    print('h0x test')
    print(h0x(np.array([1, 34.62365962451697, 78.0246928153624]), theta_optimized))

    # 6
    plot_x = [np.min(X[:, 1] - 2), np.max(X[:, 2] + 2)]
    plot_y = -1 / theta_optimized[2]*(theta_optimized[0] + np.dot(theta_optimized[1], plot_x))
    mask = y.flatten() == 1
    adm = plt.scatter(X[mask][:, 1], X[mask][:, 2])
    not_adm = plt.scatter(X[~mask][:, 1], X[~mask][:, 2])
    decision_boundary = plt.plot(plot_x, plot_y)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))
    plt.show()

    # 7
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'ex2data2.txt')
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :-1]  # first 2 column
    y = data.iloc[:, 2]  # last column
    data.head()

    # 8
    passed = y == 1
    failed = y != 1
    psd = plt.scatter(X[passed][0].values, X[passed][1].values)
    not_psd = plt.scatter(X[failed][0].values, X[failed][1].values)
    plt.xlabel('Test 1 score')
    plt.ylabel('Test 2 score')
    plt.legend((psd, not_psd), ('Passed', 'Failed'))
    plt.show()

    # 9
    map = {}
    for i in range(0, 7):
        for j in range(0, 7):
            if i + j <= 6:
                [key, fn] = polynom_multi_var(i, j)
                map[key] = fn

    # len(map.keys()) == 28
    XX = []
    for i in X.values:
        a = []
        for key in map.keys():
            a.append(map[key](i))
        XX.append(np.array(a))
    X = np.array(XX)

    # 10
    # Set regularization parameter lambda to 1
    lambda_ = 0.1
    (m, n) = X.shape
    theta = np.zeros((n + 1, 1))
    X = np.hstack((np.ones((m, 1)), X))
    y = y[:, np.newaxis]
    print('Cost at initial theta (zeros): %s', cost_function_regularized(theta, X, y, lambda_)[0][0])
    print('Expected cost (approx): 0.693')

    output = optimize.fmin_tnc(
        func=cost_function_regularized,
        x0=theta.flatten(),
        fprime=gradient_regularized,
        args=(X, y.flatten(), lambda_)
    )

    temp = output[0]
    print('Reg fmin_tnc: %s' % temp)  # theta contains the optimized values

    # 11
    temp = optimize.minimize(cost_function_regularized, theta.flatten(), (X, y.flatten(), lambda_), method='Nelder-Mead')
    print('Nelder-Mead: %s' % temp.x)

    theta_optimized = optimize.fmin_bfgs(
        cost_function_regularized,
        theta.flatten(),
        gradient_regularized,
        (X, y.flatten(), lambda_)
    )
    print('Brovden Fletcher Goldfarb Shanno alghoritm: %s' % theta_optimized)

    # 12
    print(h0x(X[0], theta_optimized))
    print(h0x(X[0], temp.x))
    print(h0x(X[0], output[0]))

    # 13
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(u), len(v)))

    for i in range(len(u)):
        for j in range(len(v)):
            a = [1]
            for key in map.keys():
                a.append(map[key]([u[i], v[j]]))
            z[i, j] = h0x(np.array(a), theta_optimized)
    mask = y.flatten() == 1
    X = data.iloc[:, :-1]
    passed = plt.scatter(X[mask][0], X[mask][1])
    failed = plt.scatter(X[~mask][0], X[~mask][1])
    plt.contour(u, v, z, 0)
    plt.xlabel('Test 1 Score')
    plt.ylabel('Test 2 Score')
    plt.legend((passed, failed), ('Passed', 'Failed'))
    plt.show()

    # 14
    # TODO: implement charts for different lambda
    X = np.array(XX)
    X = np.hstack((np.ones((m, 1)), X))
    (m, n) = X.shape
    correct_identified = 0
    for i in range(m):
        if round(h0x(X[i], theta_optimized)) == y[i]:
            correct_identified += 1
    print("Correct recognition(%): ", correct_identified / m)

    # 15
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'ex2data3.mat')
    data = sio.loadmat(file_path)
    X = data.get('X')
    y = data.get('y')

    # 16
    images = {}
    for i in range(len(y)):
        images[y[i][0]] = i  # assign the latest index of number image
    keys = images.keys()

    fig, axis = plt.subplots(1, 10)

    for j in range(len(keys)):
        # reshape back to 20 pixel by 20 pixel
        axis[j].imshow(X[images.get(images.keys()[j]), :].reshape(20, 20, order="F"), cmap="hot")
        axis[j].axis("off")

    plt.show()

    # 17
    m = len(y)
    X = np.hstack((np.ones((m, 1)), X))
    (m, n) = X.shape
    lmbda = 0.1
    k = 10
    theta = np.zeros((k, n))  # initial parameters
    print("Cost with zeros theta: ", cost_function_regularized(theta[0], X, y))
    print("Gradient with zeros theta: ", gradient_regularized(theta.T, X, y))

    # 18
    print("Cost with zeros theta: ", cost_function_regularized(theta[0], X, y, 0.001))
    print("Gradient with zeros theta: ", gradient_regularized(theta.T, X, y, 0.001))

    # 19
    for i in range(k):
        digit_class = i if i else 10
        theta[i] = optimize.fmin_cg(
            f=cost_function_regularized,
            x0=theta[i],
            fprime=gradient_regularized,
            args=(X, (y == digit_class).flatten().astype(np.int), lmbda),
            maxiter=50
        )

    # 20
    print("Predicted number: ", predict_number(X[1490], theta), "Real: ", y[1490][0])

    # 21
    pred = np.argmax(np.dot(X, theta.T), axis=1)
    pred = [e if e else 10 for e in pred]  # convert 0 to 10
    predictions = 0
    for i in range(len(pred)):
        if pred[i] == y[i][0]:
            predictions += 1

    print("Accuracy: ", (predictions / len(y)) * 100)
