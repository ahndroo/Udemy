import numpy as np
import matplotlib.pyplot as plt

from util import getData, softmax, cost2, y2indicator, error_rate, relu
from sklearn.utils import shuffle
import sys
sys.path.append('/home/ahndroo/Documents/machineLearning/Algorithms/NeuralNets')
import neuralnet as nn

def main():

    X, Y = getData()
    X, Y = shuffle(X, Y)
    K = len(np.unique(Y))
    N = len(Y)
    T = np.zeros((N, K))

    for i in range(N):
        T[i, Y[i]] = 1 # one hot encoding for targets

    batch_sz = 500
    learning_rate = [10e-5, 10e-6, 10e-7, 10-8, 10e-9]
    num_batches = len(learning_rate)
    trainCost = []; validCost = []; accValid = []; accTrain = [];
    for i in range(num_batches) :
        m = nn.NeuralNetwork(numHiddenLayer=1,numHiddenUnits=200,actFunc="Tanh")
        trainCost[i], validCost[i], accTrain[i], accValid[i] = m.train(X, T, epochs=10000, learning_rate=10e-7, reg=10e-7)

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
