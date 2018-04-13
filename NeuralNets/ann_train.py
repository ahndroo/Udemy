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
Ttrain = y2indicator(Ytrain,K) #one hot encoded targets

Xvalid = X[-100:]
Yvalid = Y[-100:]
Tvalid = y2indicator(Yvalid,K)

m = nn.NeuralNetwork(numHiddenLayer=5, numHiddenUnits=4)
trainCost, validCost = m.train(Xtrain, Ttrain, learning_rate=.001)
# Use test data
pYtrain = m.forward(Xtrain, Ttrain) # pass data through last output from NN on train data
pYvalid = m.forward(Xvalid, Xvalid) # pass validation data through NN

print("Final Train Accuracy: {}".format(m.accuracy(pYtrain, Ttrain)))
print("Final Validation Accuracy: {}".format(m.accuracy(pYvalid, Tvalid)))

legend1, = plt.plot(trainCost, label='train costs')
legend2, = plt.plot(validCost, label='validation costs')
plt.legend([legend1, legend2]); plt.show()
