<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt
from util import get_data as mnist
from datetime import datetime

def get_data():
    w = np.array([-.5, .5])
    b = 0.1
    X = np.random.random((300,2))*2-1
    Y = np.sign(X.dot(w) + b)
    return X, Y

class Perceptron:
    def fit(self, X, Y, learning_rate=1.0, epochs=1000):
        D = X.shape[1]
        self.w = np.random.randn(D)
        self.b = 0

        N = len(Y)
        costs = []

        for epoch in range(epochs):
            Yhat = self.predict(X)
            incorrect = np.nonzero(Y != Yhat)[0]
            if len(incorrect) == 0:
                break

            i = np.random.choice(incorrect)
            self.w += learning_rate*Y[i]*X[i]
            self.b += learning_rate*Y[i]

            c = len(incorrect)/float(N)
            costs.append(c)
        print('final w: {} final b: {} total epochs: {} epoch: {}'.format(self.w, self.b,epochs,epoch+1))
        plt.plot(costs)
        plt.show()

    def predict(self, X):
        return np.sign(X.dot(self.w) + self.b)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P==Y)

if __name__ == '__main__':
    X, Y = get_data()
    plt.scatter(X[:,0],X[:,1], c=Y, s=100, alpha = 0.5)
    plt.show()

    Ntrain = int(len(Y)/2)
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = Perceptron()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)

    print("Training time ",datetime.now()-t0)

    t0 = datetime.now()
    print("Train accuracy: ", model.score(Xtrain, Ytrain))
    print("Time to compute train accuracy: ", datetime.now()-t0, "\nTrain size: ", len(Ytrain))

    t0 = datetime.now()
    testScore = model.score(Xtest,  Ytest)
    print(" Test accuracy: ", testScore)
    print("Time to compute test accuracy: ", datetime.now()-t0, "\nTest size: ", len(Ytest), "\n")
=======
import numpy as np
import matplotlib.pyplot as plt
from util import get_data as mnist
from datetime import datetime

def get_data():
    w = np.array([-.5, .5])
    b = 0.1
    X = np.random.random((300,2))*2-1
    Y = np.sign(X.dot(w) + b)
    return X, Y

class Perceptron:
    def fit(self, X, Y, learning_rate=1.0, epochs=1000):
        D = X.shape[1]
        self.w = np.random.randn(D)
        self.b = 0

        N = len(Y)
        costs = []

        for epoch in range(epochs):
            Yhat = self.predict(X)
            incorrect = np.nonzero(Y != Yhat)[0]
            if len(incorrect) == 0:
                break

            i = np.random.choice(incorrect)
            self.w += learning_rate*Y[i]*X[i]
            self.b += learning_rate*Y[i]

            c = len(incorrect)/float(N)
            costs.append(c)
        print('final w: {} final b: {} total epochs: {} epoch: {}'.format(self.w, self.b,epochs,epoch+1))
        plt.plot(costs)
        plt.show()

    def predict(self, X):
        return np.sign(X.dot(self.w) + self.b)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P==Y)

if __name__ == '__main__':
    X, Y = get_data()
    plt.scatter(X[:,0],X[:,1], c=Y, s=100, alpha = 0.5)
    plt.show()

    Ntrain = int(len(Y)/2)
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = Perceptron()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)

    print("Training time ",datetime.now()-t0)

    t0 = datetime.now()
    print("Train accuracy: ", model.score(Xtrain, Ytrain))
    print("Time to compute train accuracy: ", datetime.now()-t0, "\nTrain size: ", len(Ytrain))

    t0 = datetime.now()
    testScore = model.score(Xtest,  Ytest)
    print(" Test accuracy: ", testScore)
    print("Time to compute test accuracy: ", datetime.now()-t0, "\nTest size: ", len(Ytest), "\n")
>>>>>>> 6b00251a86bc52850aa5b4d784341059d3aa9b4e
