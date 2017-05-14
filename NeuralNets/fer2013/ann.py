import numpy as np
import matplotlib.pyplot as plt

from util import getData, softmax, cost2, y2indicator, error_rate, relu
from sklearn.utils import shuffle
import sys
sys.path.append('/home/ahndroo/Documents/machineLearning/Algorithms/NeuralNets')
import neuralnet as nn

class ANN(object):
    def __init__(self, M):
        self.M = M

    # learning rate 10e-6 is too large
    def fit(self, X, Y, learning_rate=10e-7, reg=10e-7, epochs=10000, show_fig=False):
        X, Y = shuffle(X, Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        # Tvalid = y2indicator(Yvalid)
        X, Y = X[:-1000], Y[:-1000]

        N, D = X.shape
        K = len(set(Y))
        T = y2indicator(Y)
        self.W1 = np.random.randn(D, self.M) / np.sqrt(D)
        self.b1 = np.zeros(self.M)
        self.W2 = np.random.randn(self.M, K) / np.sqrt(self.M)
        self.b2 = np.zeros(K)

        costs = []
        best_validation_error = 1
        for i in range(epochs):
            # forward propagation and cost calculation
            pY, Z = self.forward(X)

            # gradient descent step
            pY_T = pY - T
            self.W2 -= learning_rate*(Z.T.dot(pY_T) + reg*self.W2)
            self.b2 -= learning_rate*(pY_T.sum(axis=0) + reg*self.b2)
            # dZ = pY_T.dot(self.W2.T) * (Z > 0) # relu
            dZ = pY_T.dot(self.W2.T) * (1 - Z*Z) # tanh
            self.W1 -= learning_rate*(X.T.dot(dZ) + reg*self.W1)
            self.b1 -= learning_rate*(dZ.sum(axis=0) + reg*self.b1)

            if i % 10 == 0:
                pYvalid, _ = self.forward(Xvalid)
                c = cost2(Yvalid, pYvalid)
                costs.append(c)
                e = error_rate(Yvalid, np.argmax(pYvalid, axis=1))
                print("i:", i, "cost:", c, "error:", e)
                if e < best_validation_error:
                    best_validation_error = e
        print("best_validation_error:", best_validation_error)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        # Z = relu(X.dot(self.W1) + self.b1)
        Z = np.tanh(X.dot(self.W1) + self.b1)
        return softmax(Z.dot(self.W2) + self.b2), Z

    def predict(self, X):
        pY, _ = self.forward(X)
        return np.argmax(pY, axis=1)

    def score(self, X, Y):
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)



def main():
    X, Y = getData()
    X, Y = shuffle(X, Y)
    K = len(np.unique(Y))
    N = len(Y)
    T = np.zeros((N, K))
    for i in range(N):
        T[i, Y[i]] = 1 # one hot encoding for targets

    m = nn.NeuralNetwork(numHiddenLayer=1,numHiddenUnits=200,actFunc="LeakyReLu")
    trainCost, validCost, accTrain, accValid = m.train(X, T, epochs=10000, learning_rate=10e-7, reg=10e-7)

    print("Final Train Accuracy {}".format(accTrain))
    print("Final Valid Accuracy {}".format(accValid))
    legend1, = plt.plot(trainCost, label='training error')
    legend2, = plt.plot(validCost, label='validation error')
    plt.legend([legend1, legend2])
    plt.show()

    # model = ANN(200)
    # model.fit(X, Y, show_fig=True)
    # print(model.score(X, Y))

if __name__ == '__main__':
    main()
