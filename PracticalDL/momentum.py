import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import get_normalized_data, error_rate, cost, y2indicator
from mlp import forward, derivative_w2, derivative_b2, derivative_w1, derivative_b1


def main():
    max_iter = 20 # make 30 for sigmoid
    print_period = 10
    
    X, Y = get_normalized_data()
    lr = 0.00004
    reg = 0.01

    Xtrain = X[:-1000,]
    Ytrain = Y[:-1000]
    Xtest  = X[-1000:,]
    Ytest  = Y[-1000:]

    N, D = Xtrain.shape
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)
    batch_sz = 500
    n_batches = N / batch_sz

    M = 300
    K = 10

    W1 = np.random(D, M) / 28
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(M)
    b2 = np.zeros(K)

    LL_batch = []
    CR_batch = []

    for i in range(max_iter):
        for j in range(n_batches):
            xBatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),:]
            yBatch = Ytrain[j*batch_sz:(j*batch_sz + batch_sz),:]
            pYbatch, Z= forward(xBatch, W1, b1, W2, b2)

            W1 -= lr*(derivative_w2(Z, yBatch, pYbatch) + reg*W2)
            b1 -= lr*(derivative_b2(yBatch, pYbatch) + reg*b2)
            W2 -= lr*(derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1)
            b2 -= lr*(derivative_b1(Z, yBatch, pYbatch, W2) + reg*b1)

            if j % print_period == 0:
                # calculate just for LL
                pY, _ = forward(Xtest, W1, b1, W2, b2)
                # print "pY:", pY
                ll = cost(pY, Ytest_ind)
                LL_batch.append(ll)
                print "Cost at iteration i=%d, j=%d: %.6f" % (i, j, ll)

                err = error_rate(pY, Ytest)
                CR_batch.append(err)
                print "Error rate:", err

    pY, _ = forward(Xtest, W1, b1, W2, b2)
print "Final error rate:", error_rate(pY, Ytest)



    x1 = np.linspace(0, 1, len(LL))
    plt.plot(x1, LL, label="full")
    x2 = np.linspace(0, 1, len(LL_stochastic))
    plt.plot(x2, LL_stochastic, label="stochastic")
    x3 = np.linspace(0, 1, len(LL_batch))
    plt.plot(x3, LL_batch, label="batch")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()
