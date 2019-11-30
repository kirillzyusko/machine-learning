from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import os
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def load_data(bit_depth, row_limit = 15000):
    ch = []
    resp = []

    with open(os.path.join(os.path.dirname(__file__), 'data', f'Base{bit_depth}.txt'), 'r') as fp:
        for idx, line in enumerate(fp):
            if idx >= row_limit:
                break
            data = line.strip().split(' ')
            ch.append(np.asarray(list(data[0]), dtype=np.int8))
            resp.append(np.asarray(data[1], dtype=np.int8))

    X = np.asarray(ch)
    y = np.array(resp)

    return X, y

def der_challenge(challenges):
    challenges_der = np.zeros(challenges.shape)
    challenges = 1 - 2 * challenges

    for i in range(len(challenges)):
        challenge = challenges[i]

        challenges_der[i][0] = challenge[0]

        for j in range(1, len(challenge)):
            challenges_der[i][j] = challenges_der[i][j-1] * challenge[j]

    return challenges_der

X, y = load_data(32)
X_der = der_challenge(X)
np.unique(y, return_counts=True)
X_train, X_test, y_train, y_test = train_test_split(X_der, y)

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_test_predicted = lr_model.predict(X_test)

print('Accuracy: ', accuracy_score(y_test, y_test_predicted))
print('F1: ', f1_score(y_test, y_test_predicted))

def get_model_errors(X_train, y_train, X_test, y_test):
    bit_depth = X_train.shape[1]
    lr_model = LogisticRegression()
    svm_model = SVC()
    nn_model = MLPClassifier(
        solver='lbfgs',
        alpha=1e-5,
        hidden_layer_sizes=(int(bit_depth * 2), int(bit_depth * 1.5)),
        random_state=1
    )

    lr_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)
    nn_model.fit(X_train, y_train)

    errors = [
        accuracy_score(y_test, lr_model.predict(X_test)),
        accuracy_score(y_test, svm_model.predict(X_test)),
        accuracy_score(y_test, nn_model.predict(X_test))
    ]

    return np.array(errors)

def show_data_count_dependency(bit_depth, data_sizes):
    X, y = load_data(bit_depth)
    X_der = der_challenge(X)
    accuracy_data = []
    for training_size in data_sizes:
        training_size = int(training_size)
        X_train = X_der[:training_size]
        y_train = y[:training_size]
        X_test = X_der[training_size:training_size + 3000]
        y_test = y[training_size:training_size + 3000]

        errors = get_model_errors(X_train, y_train, X_test, y_test)
        accuracy_data.append(np.concatenate([np.array([training_size]), errors], axis=None))

    accuracy_data = np.array(accuracy_data)

    plt.figure(figsize=(12,6))

    plt.plot(accuracy_data[:,0], accuracy_data[:,1], marker='.', color='green', label='Logistic Regression')
    plt.plot(accuracy_data[:,0], accuracy_data[:,2], marker='.', color='blue', label='SVM')
    plt.plot(accuracy_data[:,0], accuracy_data[:,3], marker='.', color='orange', label='Multilayer neural network')

    plt.title(f'Bit depth: {bit_depth}')
    plt.xticks(np.linspace(np.min(data_sizes), np.max(data_sizes), 10), fontsize=10)

    plt.xlabel('Dataset size', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

show_data_count_dependency(32, np.linspace(50, 1800, 40))
show_data_count_dependency(128, np.linspace(100, 5000, 70))

def show_N_count_dependency(files):
    accuracy_data = []

    for file in files:
        X, y = load_data(file)
        X_der = der_challenge(X)

        training_size = 10000
        X_train, X_test, y_train, y_test = train_test_split(X_der[:training_size], y[:training_size])

        errors = get_model_errors(X_train, y_train, X_test, y_test)
        accuracy_data.append(errors)

    accuracy_data = np.array(accuracy_data)

    plt.figure(figsize=(12,6))

    x = np.linspace(1, len(files), len(files))

    plt.plot(x, accuracy_data[:,0], marker='.', color='green', label='Logistic Regression')
    plt.plot(x, accuracy_data[:,1], marker='.', color='blue', label='SVM')
    plt.plot(x, accuracy_data[:,2], marker='.', color='orange', label='Multilayer neural network')

    plt.xticks(x,files, fontsize=14)

    plt.xlabel('N', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

files = [8, 16, 24, 32, 48, 64, 96, 128]
show_N_count_dependency(files)