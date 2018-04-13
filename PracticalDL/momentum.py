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

    # 1. batch GD
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
                print("Cost at iteration i=%d, j=%d: %.6f", % (i, j, ll))

                err = error_rate(pY, Ytest)
                CR_batch.append(err)
                print("Error rate:", err)

    pY, _ = forward(Xtest, W1, b1, W2, b2)
    print()"Final error rate:", error_rate(pY, Ytest))

    # 2. batch GD w/ momentum
    W1 = np.random(D, M) / 28
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(M)
    b2 = np.zeros(K)

    LL_momentum = []
    CR_momentum = []

    mu = 0.9
    dW2 = 0
    db2 = 0
    dW1 = 0
    db1 = 0

    for i in range(max_iter):
        for j in range(n_batches):
            xBatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),:]
            yBatch = Ytrain[j*batch_sz:(j*batch_sz + batch_sz),:]
            pYbatch, Z= forward(xBatch, W1, b1, W2, b2)

            # updates
            dW2 = mu*dW2 - lr*(derivative_w2(Z,yBatch, pYbatch) + reg*W2)
            W1 += dW2
            db2 = mu*db2 - lr*(derivative_b2(yBatch, pYbatch) + reg*b2)
            b2 += db2
            dW1 = mu*dW1 - lr*(derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1)
            W1 += dW1
            db1 = mu*db1 - lr*(derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1)
            b1 += db1


            if j % print_period == 0:
                # calculate just for LL
                pY, _ = forward(Xtest, W1, b1, W2, b2)
                # print "pY:", pY
                ll = cost(pY, Ytest_ind)
                LL_momentum.append(ll)
                print("Cost at iteration i=%d, j=%d: %.6f", % (i, j, ll))

                err = error_rate(pY, Ytest)
                CR_momentum.append(err)
                print("Error rate:", err)
    pY, _ = forward(Xtest, W1, b1, W2, b2)
    print()"Final error rate:", error_rate(pY, Ytest))

    # 3. batch GD w/ Nesterov momentum
    W1 = np.random(D, M) / 28
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(M)
    b2 = np.zeros(K)

    LL_nest = []
    CR_nest = []

    mu = 0.9
    # dW2 = 0
    # db2 = 0
    # dW1 = 0
    # db1 = 0
    vW2 = 0
    vb2 = 0
    vW1 = 0
    vb1 = 0

    for i in range(max_iter):
        for j in range(n_batches):
            # because we want g(t) = grad(f(W(t-1) - lr*mu*dW(t-1)))
            # dW(t) = mu*dW(t-1) + g(t)
            # W(t) = W(t-1) - mu*dW(t)
            W1_tmp = W1 - lr*mu*vW1
            b1_tmp = b1 - lr*mu*vb1
            W2_tmp = W2 - lr*mu*vW2
            b2_tmp = b2 - lr*mu*vb2

            xBatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),:]
            yBatch = Ytrain[j*batch_sz:(j*batch_sz + batch_sz),:]

            pYbatch, Z= forward(xBatch, W1_tmp, b1_tmp, W2_tmp, b2_tmp)

            # updates
            vW2 = mu*vW2 + derivative_w2(Z, Ybatch, pYbatch) + reg*W2_tmp
            W2 -= lr*vW2
            vb2 = mu*vb2 + derivative_b2(Ybatch, pYbatch) + reg*b2_tmp
            b2 -= lr*vb2
            vW1 = mu*vW1 + derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2_tmp) + reg*W1_tmp
            W1 -= lr*vW1
            vb1 = mu*vb1 + derivative_b1(Z, Ybatch, pYbatch, W2_tmp) + reg*b1_tmp
            b1 -= lr*vb1

            if j % print_period == 0:
                # calculate just for LL
                pY, _ = forward(Xtest, W1, b1, W2, b2)
                # print "pY:", pY
                ll = cost(pY, Ytest_ind)
                LL_nest.append(ll)
                print("Cost at iteration i=%d, j=%d: %.6f", % (i, j, ll))

                err = error_rate(pY, Ytest)
                CR_nest.append(err)
                print("Error rate:", err)
    pY, _ = forward(Xtest, W1, b1, W2, b2)
    print()"Final error rate:", error_rate(pY, Ytest))



    plt.plot(LL_batch, label="batch")
    plt.plot(LL_momentum, label="momentum")
    plt.plot(LL_nest, label="nesterov")
    plt.legend()
    plt.show()



if __name__ == '__main__':
main()