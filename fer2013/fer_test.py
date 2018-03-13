import numpy as np
import matplotlib.pyplot as plt
import cnn_tf as cnn
import tensorflow as tf
from util import getImageData

X, Y = getImageData()

X = X.transpose(0,2,3,1).astype(np.float32)

learning_rates = [10e-3, 10e-4, 10e-5, 10e-6, 10e-7, 10e-8] # best learning rate is 10e-8
# learing_rate = 10e-8

# convLayer_sz = [[(10, 5, 5), (10, 5, 5)], [(20, 5, 5), (20, 5, 5)],
#                 [(30, 5, 5), (30, 5, 5)], [(40, 5, 5), (40, 5, 5)]]
convLayer_sz = [(10, 5, 5), (10, 5, 5)]
mlpLayer_sz = [500, 300]

costs = []
acc = []

file = open('Validation Data Set Scores.txt','w')
file.write('Validation results:\n')

for j, lr in enumerate(learning_rates):
    # for i, cp in enumerate(convLayer_sz):
    model = cnn.CNN(convpool_sz = convLayer_sz, hidden_sz = mlpLayer_sz)
    c, a = model.train(X, Y, learning_rate=lr, epochs=500, batch_sz=100, dispFig=False)
    costs.append(c); acc.append(a)
    x = len(costs[j])
    fig = plt.figure(); fig.suptitle('Learning Rate' + str(lr))
    plt.subplot(3,3,j+1); plt.plot(costs[j]); plt.xlabel('Batch #'); plt.ylabel('Cost');plt.title('Feature Map Depth: ' + str(convLayer_sz[0]))
    fig.savefig('test'+str(lr)+'.jpg')
    file.write('ConvPool Size: ' + str(convPool_sz[0]) + 'Learning Rate: ' + str(lr) + 'Final Accuracy: ' + str(a) + '\n')

file.close()
