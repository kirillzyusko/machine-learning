from __future__ import division
from goto import with_goto
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.svm import SVC
import re
from nltk.stem import PorterStemmer


def decision_boundary(classifier, X, y, xlim_min=0, xlim_max=4.5, ylim_min=1.5, ylim_max=5):
    m, n = X.shape[0], X.shape[1]
    pos, neg = (y == 1).reshape(m, 1).flatten(), (y == 0).reshape(m, 1).flatten()
    plt.figure(figsize=(8, 6))
    plt.scatter(X[pos, 0], X[pos, 1])
    plt.scatter(X[neg, 0], X[neg, 1], marker="x")
    # plotting the decision boundary
    X_1, X_2 = np.meshgrid(
        np.linspace(X[:, 0].min(), X[:, 1].max(), num=100),
        np.linspace(X[:, 1].min(), X[:, 1].max(), num=100)
    )
    plt.contour(X_1, X_2, classifier.predict(np.array([X_1.flatten(), X_2.flatten()]).T).reshape(X_1.shape), 1)
    plt.xlim(xlim_min, xlim_max)
    plt.ylim(ylim_min, ylim_max)

    plt.show()


def gauss_kernel_carried(sigma):
    def gauss_kernel(x1, x2):
        sigma_squared = np.power(sigma, 2)
        matrix = np.power(x1-x2, 2)

        return np.exp(-np.sum(matrix)/(2*sigma_squared))

    return gauss_kernel


def plot_data(X, y):
    m, n = X.shape[0], X.shape[1]
    pos, neg = (y == 1).reshape(m, 1).flatten(), (y == 0).reshape(m, 1).flatten()
    plt.scatter(X[pos, 0], X[pos, 1])
    plt.scatter(X[neg, 0], X[neg, 1])
    plt.show()


def dataset3Params(X, y, Xval, yval, values):
    # You need to return the following variables correctly.
    C = values[0]
    sigma = values[0]
    result_score = 0
    # ====================== YOUR CODE HERE ======================
    for i in values:
        for j in values:
            gamma = 1 / j
            classifier = SVC(C=i, gamma=gamma, kernel='rbf')
            classifier.fit(X, y)
            prediction = classifier.predict(Xval)
            score = classifier.score(Xval, yval)
            print("i: ", i, "j: ", j, "score: ", score)
            if score > result_score:
                result_score = score
                C = i
                sigma = gamma

    # ============================================================
    return C, sigma


def process_email(email_contents):
    """
    Preprocesses the body of an email and returns a list of indices of the words contained in the email.
    """
    # a - Lower case
    email_contents = email_contents.lower()

    # b - remove html/xml tags
    email_contents = re.sub("<[^>]*>", " ", email_contents).split(" ")
    email_contents = filter(len, email_contents)
    email_contents = ' '.join(email_contents)

    # c - Handle URLS
    email_contents = re.sub("[http|https]://[^\s]*", "httpaddr", email_contents)

    # d - Handle Email Addresses
    email_contents = re.sub("[^\s]+@[^\s]+", "emailaddr", email_contents)

    # e - Handle numbers
    email_contents = re.sub("[0-9]+", "number", email_contents)

    # f - Handle $ sign
    email_contents = re.sub("[$]+", "dollar", email_contents)

    # Strip all special characters
    special_chars = [
        "<", "[", "^", ">", "+", "?", "!", "'", ".", ",", ":",
        "*", "%", "#", "_", "="
    ]
    for char in special_chars:
        email_contents = email_contents.replace(str(char), "")
    email_contents = email_contents.replace("\n", " ")

    # Stem the word
    ps = PorterStemmer()
    email_contents = [ps.stem(token) for token in email_contents.split(" ")]
    email_contents = " ".join(email_contents)

    return email_contents


def find_word_indices(processed_email, vocabList_d):
    # Process the email and return word_indices

    word_indices = []

    for char in processed_email.split():
        if len(char) > 1 and char in vocabList_d:
            word_indices.append(int(vocabList_d[char]))

    return word_indices


def transform_email_to_features(email_contents, vocabList_d):
    processed_email = process_email(email_contents)
    word_indices = find_word_indices(processed_email, vocabList_d)
    features = email_features(word_indices, vocabList_d)

    return features


def email_features(word_indices, vocabList_d):
    """
    Takes in a word_indices vector and  produces a feature vector from the word indices.
    """
    n = len(vocabList_d)

    features = np.zeros((n, 1))

    for i in word_indices:
        features[i] = 1

    return features


@with_goto
def main():
    goto .task
    # 1
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'ex5data1.mat')
    dataset = sio.loadmat(file_path)
    X = dataset["X"]
    y = dataset["y"]

    # 2
    # [:, 0] - equals to flatten
    plot_data(X, y)

    # 3
    classifier = SVC(kernel="linear")
    classifier.fit(X, y.flatten())  # default C=1

    # 4
    decision_boundary(classifier, X, y)
    # Test C = 100
    classifier2 = SVC(C=100, kernel="linear")  # gives a decision boundary that overfits the training examples
    classifier2.fit(X, y.flatten())
    decision_boundary(classifier2, X, y)

    # 5
    x1 = np.array([1, 2, 1])
    x2 = np.array([0, 4, -1])
    sigma = 2

    sim = gauss_kernel_carried(sigma)(x1, x2)
    print('Gaussian Kernel between x1 = [1, 2, 1], x2 = [0, 4, -1], sigma = %0.2f:'
          '\n\t%f\n(for sigma = 2, this value should be about 0.324652)\n' % (sigma, sim))

    # 6
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'ex5data2.mat')
    dataset = sio.loadmat(file_path)
    X = dataset["X"]
    y = dataset["y"]

    plot_data(X, y)

    # 7
    sigma = 0.1
    kernel = gauss_kernel_carried(sigma)

    # 8
    gamma = np.power(sigma, -2.)
    classifier3 = SVC(C=1, kernel='rbf', gamma=gamma)
    classifier3.fit(X, y.flatten())

    # 9
    decision_boundary(classifier3, X, y, 0, 1, .4, 1)

    # 10
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'ex5data3.mat')
    dataset = sio.loadmat(file_path)
    X = dataset["X"]
    y = dataset["y"]
    Xval = dataset["Xval"]
    yval = dataset["yval"]

    plot_data(X, y)
    # plot_data(Xval, yval)

    # 11
    vals = [0.01, 0.03, 0.1, 0.3, 0.5, 1, 3, 10, 30, 50, 100]
    C, gamma = dataset3Params(X, y.flatten(), Xval, yval.flatten(), vals)
    print("C: ", C, ", gamma: ", gamma)
    classifier4 = SVC(C=C, gamma=gamma, kernel='rbf')
    classifier4.fit(X, y.flatten())

    # 12
    decision_boundary(classifier4, Xval, yval, -0.6, 0.3, -0.7, 0.7)

    # 13
    label .task
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'spamTrain.mat')
    dataset = sio.loadmat(file_path)
    X = dataset["X"]
    y = dataset["y"]

    # 14
    C = 0.1
    classifier5 = SVC(C=C, kernel='linear')
    classifier5.fit(X, y.flatten())
    print('Training Accuracy: ', (classifier5.score(X, y.flatten())) * 100)

    # 15
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'spamTest.mat')
    dataset = sio.loadmat(file_path)
    Xtest = dataset["Xtest"]
    ytest = dataset["ytest"]

    # 16
    print("Test Accuracy (linear):", (classifier5.score(Xtest, ytest.flatten()))*100, "%")
    # C = 1, gamma=0.01 ~98.7
    # C = 30, gamma=0.001 ~ 99.1
    classifier6 = SVC(C=30, kernel='rbf', gamma=0.001)
    classifier6.fit(X, y.flatten())
    print("Training Accuracy (gaussian):", (classifier6.score(X, y.flatten()))*100, "%")
    print("Test Accuracy (gaussian):", (classifier6.score(Xtest, ytest.flatten()))*100, "%")


    # 17
    # simple test
    assert process_email("Hello WoRlD") == "hello world", 'Not implemented toLowerCase functional'
    assert process_email("<i class=\"italic\"><p><b>Hello world</b><span>Ola</span></p></i>") == "hello world ola", 'Not implemented removing html-tags functional'
    assert process_email("http://www.leningrad.spb.ru") == "htthttpaddr", 'Not implemented replacing http addresses functional'
    assert process_email("zyusko.kirik@gmail.com") == "emailaddr", 'Not implemented replacing email addresses functional'
    assert process_email("amount is 5334$") == "amount is numberdollar", 'Not implemented: replacing $ functional'
    assert process_email("discounted") == "discount", 'Stemming not implemented'
    assert process_email("*you won* ^_^") == "you won ", 'Removing special characters not implemented'

    # 18
    vocabList = open(os.path.join(os.path.dirname(__file__), 'data', 'vocab.txt'), "r").read()
    vocabList = vocabList.split("\n")
    vocabList_d = {}
    for ea in vocabList:
        [value, key] = ea.split("\t")
        vocabList_d[key] = value

    # 19
    processed_email = process_email("Content of email")
    print(processed_email)
    word_indices = find_word_indices(processed_email, vocabList_d)
    print(word_indices)

    # 20
    features = email_features(word_indices, vocabList_d)
    print(features.flatten())

    # 21
    email_sample1 = open(os.path.join(os.path.dirname(__file__), 'data', 'emailSample1.txt'), "r").read()
    email_sample2 = open(os.path.join(os.path.dirname(__file__), 'data', 'emailSample2.txt'), "r").read()
    spam_sample1 = open(os.path.join(os.path.dirname(__file__), 'data', 'spamSample1.txt'), "r").read()
    spam_sample2 = open(os.path.join(os.path.dirname(__file__), 'data', 'spamSample2.txt'), "r").read()

    email_sample1 = transform_email_to_features(email_sample1, vocabList_d)
    email_sample2 = transform_email_to_features(email_sample2, vocabList_d)
    spam_sample1 = transform_email_to_features(spam_sample1, vocabList_d)
    spam_sample2 = transform_email_to_features(spam_sample2, vocabList_d)

    print('Spam  -> 1\nEmail -> 0')
    print('Linear Kernel: ')
    print('False', classifier5.predict(email_sample1.T))
    print('False', classifier5.predict(email_sample2.T))
    print('Spam', classifier5.predict(spam_sample1.T))
    print('Spam', classifier5.predict(spam_sample2.T))
    print('\n Gaussian Kernel: ')
    print('False', classifier6.predict(email_sample1.T))
    print('False', classifier6.predict(email_sample2.T))
    print('Spam', classifier6.predict(spam_sample1.T))
    print('Spam', classifier6.predict(spam_sample2.T))
    print('\n')

    # 22
    spam_sample = "Hi, Kirill Zyusko, Items on your wishlist are now discounted! Some items you've added to your personal wishlist are currently discounted on http://www.gog.com/ - you can find all the details below or directly on your https://www.gog.com/account/wishlist."
    spam_sample = transform_email_to_features(spam_sample, vocabList_d)
    print('\n Gaussian Kernel: ')
    print('Spam', classifier6.predict(spam_sample.T))

    email_sample = "Hi Kirill Zyusko, Your job search status is open, but not actively looking, but we need more information before accelerating your matches. Once you tell us just a little bit more about yourself, we'll get you in front of companies and send you any new jobs that match your interests."
    email_sample = transform_email_to_features(email_sample, vocabList_d)
    print('Email (should be false - at least gmail doesn\'t classify this message as a spam)', classifier6.predict(email_sample.T))

    spam_sample3 = "Hi, Hope you are doing great. This is Josh from Avco Consulting Inc. Inc is a global IT company based in Worcester, MA. Our leadership in the industry has been established by our excellence in helping clients use Information Technology to achieve their business objectives. Our core competencies are Information Technology (IT)services and Project Management. I have available consultants with Excellent communication, analytical, and team work skills. I would appreciate if you - or someone you can recommend share the suitable requirements accordingly. Below is the list of the available consultant with different skills set and their preferred locations."
    spam_sample3 = transform_email_to_features(spam_sample3, vocabList_d)
    print('Spam', classifier6.predict(spam_sample3.T))

    # 23 - 24
    # Look at utils/download.py

    # 25
    # Change #18 to vocab2.txt, #13 to myTrain.mat and #15 to myTest.mat to see the difference


if __name__ == '__main__':
    main()
