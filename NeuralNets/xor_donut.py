import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/ahndroo/Documents/machineLearning/Algorithms/NeuralNets')
import neuralnet as nn

def test_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])

    m = nn.NeuralNetwork()
    trainCost, validCost = m.train(X, Y)
    pYtrain = m.forward(Xtrain, Ttrain) # pass data through last output from NN on train data
    pYvalid = m.forward(Xvalid, Xvalid) # pass validation data through NN

    print("Final Train Accuracy: {}".format(m.accuracy(pYtrain, Ttrain)))
    print("Final Validation Accuracy: {}".format(m.accuracy(pYvalid, Tvalid)))
