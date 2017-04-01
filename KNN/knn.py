#
# This is an example of a K-Nearest Neighbors classifier on MNIST data.
# Attempt K=1..5 to show how to determine the best K.

import numpy as np
import matplotlib.pyplot as plt
from sortedcontainers import SortedList
from util import get_data
from datetime import datetime

class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y = np.zeros(len(X))
        for i, x in enumerate(X):
            sl = SortedList(load=self.k)
            for j, xt in enumerate(self.X):
                diff = x-xt
                d = diff.dot(diff)
                if len(sl) < self.k:
                     # if sl has less items than defined k, begin to populate sl
                    sl.add((d, self.y[j]))
                else:
                    # sl is filled with k values, if new value scores lower than
                    # last value in sl then replace
                    if d < sl[-1][0]:
                        del sl[-1]
                        sl.add((d, self.y[j]))
            votes = {}
            for _, v in sl:
                # get() method takes in key and default value if key not found
                # zero in this case
                votes[v] = votes.get(v, 0) + 1
            max_votes = 0
            max_votes_class = -1
            for v, count in votes.items():
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
            y[i] = max_votes_class
        return y

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P==Y)


if __name__ == '__main__':
    X, Y = get_data(2000)
    Ntrain = 1000
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
    scores = []
    for k in (3,5,8,10,15,20):
        print("For k:", k)
        knn = KNN(k)
        t0 = datetime.now()
        knn.fit(Xtrain, Ytrain)
        print("Training time ",datetime.now()-t0)

        t0 = datetime.now()
        print("Train accuracy: ", knn.score(Xtrain, Ytrain))
        print("Time to compute train accuracy: ", datetime.now()-t0, "\nTrain size: ", len(Ytrain))

        t0 = datetime.now()
        testScore = knn.score(Xtest,  Ytest)
        print(" Test accuracy: ", testScore)
        print("Time to compute test accuracy: ", datetime.now()-t0, "\nTest size: ", len(Ytest), "\n")
        scores.append(testScore)
    plt.plot(scores)
    plt.show()
