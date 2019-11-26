from goto import with_goto
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds


def cost_function(params, Y, R, num_users, num_movies, num_features, Lambda):
    """
    Returns the cost and gradient for the collaborative filtering problem
    """

    # Unfold the params
    X = params[:num_movies*num_features].reshape(num_movies,num_features)
    Theta = params[num_movies*num_features:].reshape(num_users,num_features)

    predictions = np.dot(X, Theta.T)
    err = (predictions - Y)
    J = 1/2 * np.sum((err**2) * R)

    # compute regularized cost function
    reg_X = Lambda/2 * np.sum(Theta**2)
    reg_Theta = Lambda/2 * np.sum(X**2)
    reg_J = J + reg_X + reg_Theta

    # Compute gradient
    X_grad = np.dot(err*R, Theta)
    Theta_grad = np.dot((err*R).T, X)
    grad = np.append(X_grad.flatten(), Theta_grad.flatten())

    # Compute regularized gradient
    reg_X_grad = X_grad + Lambda*X
    reg_Theta_grad = Theta_grad + Lambda*Theta
    reg_grad = np.append(reg_X_grad.flatten(), reg_Theta_grad.flatten())

    return J, grad, reg_J, reg_grad


def normalize_ratings(Y, R):
    """
    normalized Y so that each movie has a rating of 0 on average, and returns the mean rating in Ymean.
    """
    m, n = Y.shape[0], Y.shape[1]
    Ymean = np.zeros((m, 1))
    Ynorm = np.zeros((m, n))

    for i in range(m):
        Ymean[i] = np.sum(Y[i, :])/np.count_nonzero(R[i, :])
        Ynorm[i, R[i, :] == 1] = Y[i, R[i, :] == 1] - Ymean[i]

    return Ynorm, Ymean


def gradient_descent(initial_parameters,Y,R,num_users,num_movies,num_features,alpha,num_iters,Lambda):
    """
    Optimize X and Theta
    """
    # unfold the parameters
    X = initial_parameters[:num_movies*num_features].reshape(num_movies,num_features)
    Theta = initial_parameters[num_movies*num_features:].reshape(num_users,num_features)

    J_history = []

    for i in range(num_iters):
        params = np.append(X.flatten(), Theta.flatten())
        cost, grad = cost_function(params, Y, R, num_users, num_movies, num_features, Lambda)[2:]

        # unfold grad
        X_grad = grad[:num_movies*num_features].reshape(num_movies,num_features)
        Theta_grad = grad[num_movies*num_features:].reshape(num_users,num_features)
        X = X - (alpha * X_grad)
        Theta = Theta - (alpha * Theta_grad)
        J_history.append(cost)

    paramsFinal = np.append(X.flatten(), Theta.flatten())
    return paramsFinal, J_history


def get_my_dataset():
    my_ratings = np.zeros((1682, 1))
    # Set my own estimation
    my_ratings[190] = 5
    my_ratings[63] = 5
    my_ratings[70] = 4
    my_ratings[68] = 5
    my_ratings[95] = 5

    return my_ratings


@with_goto
def main():
    goto .task
    label .task
    # 1
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'ex9_movies.mat')
    dataset = sio.loadmat(file_path)

    # Y is a 1682x943 matrix, containing ratings (1-5) of
    # 1682 movies on 943 users
    # R is a 1682x943 matrix, where R(i,j) = 1
    # if and only if user j gave a rating to movie i
    Y, R = dataset['Y'], dataset['R']
    X = np.zeros((1682, 10))  # 1682 X 10 matrix , num_movies X num_features matrix of movie features
    Theta = np.zeros((943, 10))  # 943 X 10 matrix, num_users X num_features matrix of user features

    # 2
    num_users, num_movies, num_features = 4, 5, 3

    # 3-6
    X_test = X[:num_movies, :num_features]
    Theta_test = Theta[:num_users, :num_features]
    Y_test = Y[:num_movies, :num_users]
    R_test = R[:num_movies, :num_users]
    params = np.append(X_test.flatten(), Theta_test.flatten())

    # Evaluate cost function
    J, grad = cost_function(params, Y_test, R_test, num_users, num_movies, num_features, 0)[:2]
    print("Cost at loaded parameters:", J)
    J2, grad2 = cost_function(params, Y_test, R_test, num_users, num_movies, num_features, 1.5)[2:]
    print("Cost at loaded parameters (lambda = 1.5):", J2)

    # 7
    # Normalize Ratings
    Ynorm, Ymean = normalize_ratings(Y, R)
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = 10
    # Set initial Parameters (Theta,X)
    X = np.random.randn(num_movies, num_features)
    Theta = np.random.randn(num_users, num_features)
    initial_parameters = np.append(X.flatten(), Theta.flatten())
    Lambda = 10
    # Optimize parameters using Gradient Descent
    paramsFinal, J_history = gradient_descent(initial_parameters, Y, R, num_users, num_movies, num_features, 0.001, 400, Lambda)

    plt.plot(J_history)
    plt.xlabel("Iteration")
    plt.ylabel("$J(\Theta)$")
    plt.title("Cost function using Gradient Descent")

    plt.show()

    # 8
    # load movie list
    movieList = open(os.path.join(os.path.dirname(__file__), 'data', 'movie_ids.txt'), "r").read().split("\n")[:-1]
    # Initialize my ratings
    my_ratings = get_my_dataset()

    print("My ratings:\n")
    for i in range(len(my_ratings)):
        if my_ratings[i] > 0:
            print("Rated", int(my_ratings[i]), "for index", movieList[i])

    # 9
    # unfold paramaters
    X = paramsFinal[:num_movies*num_features].reshape(num_movies, num_features)
    Theta = paramsFinal[num_movies*num_features:].reshape(num_users, num_features)
    # Predict rating
    p = np.dot(X, Theta.T)
    my_predictions = p[:, 0][:, np.newaxis] + Ymean
    df = pd.DataFrame(np.hstack((my_predictions, np.array(movieList)[:, np.newaxis])))
    df.sort_values(by=[0], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("Top recommendations for you:\n")

    for i in range(10):
        print("Predicting rating", round(float(df[0][i]), 1), " for index", df[1][i])

    # 10
    Y = np.array(Y.T)
    Y = np.append(Y, [get_my_dataset().flatten()], axis=0)
    R = Y
    user_ratings_mean = np.mean(R, axis=1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)
    U, sigma, Vt = svds(R_demeaned, k=300)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

    for i in all_user_predicted_ratings[943].argsort()[-15:][::-1]:
        print("Predicting rating", all_user_predicted_ratings[943][i], "index: ", movieList[i])


if __name__ == '__main__':
    main()
