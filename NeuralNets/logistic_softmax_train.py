import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from process import get_data
import sys
sys.path.append('/home/ahndroo/Documents/machineLearning/Algorithms/NeuralNets')
import neuralnet as nn

def y2indicator(y,K):
    N = len(y)
    ind = np.zeros((N,K))
    for i in range(N):
        ind[i,y[i]] = 1
    return ind

X, Y = get_data()
X, Y = shuffle(X, Y)
Y = Y.astype(np.int32)
D = X.shape[1]
K = len(set(Y))

Xtrain = X[:-100]
Ytrain = Y[:-100]
Ytrain_ind = y2indicator(Ytrain,K)

Xtest = X[-100:]
Ytest = Y[-100:]
Ytest_ind = y2indicator(Ytest,K)

m = nn.NeuralNetwork()
m.train(Xtrain, Ytrain_ind, learning_rate=.001)
# Use test data
m.forward(Xtest)

print("Final Cost: {} Accuracy: {}".format(m.cost(), m.accuracy()))
