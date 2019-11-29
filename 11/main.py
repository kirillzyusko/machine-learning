import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import random
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings('ignore')


def into_features_vect(chall):
    "Transforms a challenge into a feature vector"
    phi = []
    for i in range(1,len(chall)):
        s = sum(chall[i:])
        if s % 2 == 0:
            phi.append(1)
        else:
            phi.append(-1)
    phi.append(1)
    return phi


class Stage:
    _delay_out_a = 0.
    _delay_out_b = 0.
    _selector = 0

    def __init__(self,delay_a,delay_b):
        self._delay_out_a = delay_a
        self._delay_out_b = delay_b

    def set_selector(self,s):
        self._selector = s

    def get_output(self,delay_in_a, delay_in_b):
        if self._selector == 0:
            return (delay_in_a  + self._delay_out_a,
                    delay_in_b  + self._delay_out_b)
        else:
            return (delay_in_b  + self._delay_out_a,
                    delay_in_a  + self._delay_out_b)


class ArbiterPUF:

    def __init__(self,n):
        self._stages = []

        for _ in range(n):
            d1 = random.random()
            d2 = random.random()
            self._stages.append(Stage(d1,d2))

    def get_output(self,chall):
        # Set challenge
        for stage,bit in zip(self._stages,chall):
            stage.set_selector(bit)

        # Compute output
        delay = (0,0)
        for s in self._stages:
            delay = s.get_output(delay[0],delay[1])

        if delay[0] < delay[1]:
            return 0
        else:
            return 1


class Stage:
    _delay_out_a = 0.
    _delay_out_b = 0.
    _selector = 0

    def __init__(self,delay_a,delay_b):
        self._delay_out_a = delay_a
        self._delay_out_b = delay_b

    def set_selector(self,s):
        self._selector = s

    def get_output(self,delay_in_a, delay_in_b):
        if self._selector == 0:
            return (delay_in_a  + self._delay_out_a,
                    delay_in_b  + self._delay_out_b)
        else:
            return (delay_in_b  + self._delay_out_a,
                    delay_in_a  + self._delay_out_b)


class ArbiterPUF:

    def __init__(self,n):
        self._stages = []

        for _ in range(n):
            d1 = random.random()
            d2 = random.random()
            self._stages.append(Stage(d1,d2))

    def get_output(self,chall):
        # Set challenge
        for stage,bit in zip(self._stages,chall):
            stage.set_selector(bit)

        # Compute output
        delay = (0,0)
        for s in self._stages:
            delay = s.get_output(delay[0],delay[1])

        if delay[0] < delay[1]:
            return 0
        else:
            return 1


N  = 32     # Size of the PUF
LS = 600    # Size learning set
TS = 10000  # Size testing set
apuf = ArbiterPUF(N)

# Creating training suite
learningX = [[random.choice([0,1]) for _ in range(N)] for _ in range(LS)] # Challenges
learningY = [apuf.get_output(chall) for chall in learningX] # Outputs PUF

# Creating testing suite
testingX = [[random.choice([0,1]) for _ in range(N)] for _ in range(TS)]
testingY = [apuf.get_output(chall) for chall in testingX]

# Convert challenges into feature vectors
learningX = [into_features_vect(c) for c in learningX]
testingX = [into_features_vect(c) for c in testingX]

# Prediction
lr = LogisticRegression()
lr.fit(learningX, learningY)
print("Score arbiter PUF (%d stages): %f" % (N,lr.score(testingX,testingY)))

svc = SVC()
svc.fit(learningX, learningY)
print("Score arbiter PUF (%d stages): %f" % (N, svc.score(testingX, testingY)))

gb = GradientBoostingClassifier()
gb.fit(learningX, learningY)
print("Score arbiter PUF (%d stages): %f" % (N, gb.score(testingX, testingY)))

learning_set_numbers = np.arange(100, 1000, 50)
scores = []

for i in learning_set_numbers:
    N  = 32     # Size of the PUF
    LS = i    # Size learning set
    TS = 10000  # Size testing set
    apuf = ArbiterPUF(N)

    # Creating training suite
    learningX = [[random.choice([0,1]) for _ in range(N)] for _ in range(LS)] # Challenges
    learningY = [apuf.get_output(chall) for chall in learningX] # Outputs PUF

    # Creating testing suite
    testingX = [[random.choice([0,1]) for _ in range(N)] for _ in range(TS)]
    testingY = [apuf.get_output(chall) for chall in testingX]

    # Convert challenges into feature vectors
    learningX = [into_features_vect(c) for c in learningX]
    testingX = [into_features_vect(c) for c in testingX]

    # Prediction
    lr = LogisticRegression()
    lr.fit(learningX, learningY)
    scores.append(lr.score(testingX, testingY))

plt.plot(learning_set_numbers, scores, LineWidth=2)
plt.ylabel('Accuracy')
plt.xlabel('N')
plt.show()