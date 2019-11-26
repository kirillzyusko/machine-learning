"""
File -> Settings -> Tools -> Python Scientific -> uncheck mark
Run with python 3.7.5 (64 bit)
"""

from __future__ import division
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from goto import with_goto
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool
from sklearn import ensemble


def gradient(Y, X, base_algorithms_list, coefficients_list):
    return Y - predict(X, base_algorithms_list, coefficients_list)


def predict(X, base_algorithms_list, coefficients_list):
    return [sum([coeff * algo.predict([x])[0] for algo, coeff in zip(base_algorithms_list, coefficients_list)])
            for x in X]


def check_quality(X_train, y_train, X_test, y_test, get_coefficient):
    def get_model(trees_number=50, depth=5):
        print(trees_number, depth)
        # 4
        base_algorithms_list = []
        coefficients_list = []

        # 5
        for i in range(0, trees_number):
            # create new algorithm
            rg = DecisionTreeRegressor(random_state=42, max_depth=depth)
            # fit algo in train dataset and new target
            rg.fit(X_train, gradient(y_train, X_train, base_algorithms_list, coefficients_list))
            # append results
            base_algorithms_list.append(rg)
            # ======= 6 =======
            coefficients_list.append(get_coefficient(i))

        # 7
        pred = predict(X_test, base_algorithms_list, coefficients_list)
        print(np.sqrt(mean_squared_error(y_test, pred)))
        return np.sqrt(mean_squared_error(y_test, pred))

    return get_model


def mock_coefficient(i):
    return 0.9


def calculate_coefficient(i):
    return 0.9 / (1.0 + i)


class DecisionTreeParams():
    def __init__(self, X_train, y_train, X_test, y_test, coefficients, number_trees, depth):
        """Constructor"""
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.coefficients = coefficients
        self.number_trees = number_trees
        self.depth = depth

    def worker(self):
        return check_quality(self.X_train, self.y_train, self.X_test, self.y_test, self.coefficients)(self.number_trees, self.depth)

    def run_gb(self):
        params = {'n_estimators': self.number_trees, 'max_depth': self.depth, 'min_samples_split': 2,
                  'learning_rate': 0.01, 'loss': 'ls'}
        clf = ensemble.GradientBoostingRegressor(**params)
        clf.fit(self.X_train, self.y_train)
        return np.sqrt(mean_squared_error(self.y_test, clf.predict(self.X_test)))


def job(A):
    return A.worker()


def job_gb(A):
    return A.run_gb()


@with_goto
def main():
    goto .task
    label .task
    # 1
    boston = datasets.load_boston()
    X, Y = boston.data, boston.target

    # 2
    # ###################### QUITE IMPORTANT ##########################
    # overall MSE really depends on factor how do we split our data
    # so changing `random_state` may lead to various results.
    # In order to have the same results I set seed to value equal to 51
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=51)

    # 3
    mocked_coefficient_quality = check_quality(X_train, y_train, X_test, y_test, mock_coefficient)()
    print(mocked_coefficient_quality)

    # 8
    sequences_coefficient_quality = check_quality(X_train, y_train, X_test, y_test, calculate_coefficient)()
    print(sequences_coefficient_quality)

    # 9
    # since Intel i7 6700HQ has 8 threads
    trees_number = [50, 100, 150, 200, 250, 300, 400, 500]
    params = []

    for i in trees_number:
        params.append(DecisionTreeParams(X_train, y_train, X_test, y_test, calculate_coefficient, i, 5))

    pool = Pool(len(trees_number))
    # change job_gb to job for getting my own implementation of gradient boosting
    errors = pool.map(job_gb, params)

    plt.plot(trees_number, errors, color='orange')
    plt.xlabel('Trees number')
    plt.ylabel('Error')
    plt.title('Error in depending trees number')
    plt.show()

    depth = [2, 4, 6, 8, 10, 13, 17, 20]
    params = []

    for i in depth:
        params.append(DecisionTreeParams(X_train, y_train, X_test, y_test, calculate_coefficient, 50, i))

    # change job_gb to job for getting my own implementation of gradient boosting
    errors = pool.map(job_gb, params)

    plt.plot(depth, errors, color='orange')
    plt.xlabel('Depth')
    plt.ylabel('Error')
    plt.title('Error in depending of trees depth')
    plt.show()

    # 10
    reg = LinearRegression().fit(X_train, y_train)
    pred = reg.predict(X_test)
    print('Linear regression MSE: ', np.sqrt(mean_squared_error(y_test, pred)))


if __name__ == '__main__':
    main()
