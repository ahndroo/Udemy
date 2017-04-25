import numpy as np
import matplotlib.pyplot as plt

def forward(X, W1, b1, W2, b2):
    Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y, Z

def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total

def derivative_w2(Z, T, Y):
    N, K = T.shape
    M = Z.shape[1]

    # slow method--direct from derivation
    # ret1 = np.zeros((M, K))
    # for n in range(N):
    #     for m in range(M):
    #         for k in range(K):
    #             ret1[m,k] += (T[n,k] - Y[n,k]) * Z[n,m]
    # remove loop over m
    # ret2 = np.zeros((M, K))
    # for n in range(N):
    #     for k in range(K):
    #         ret2[:,k] += (T[n, k] - Y[n, k]) * Z[n, :]
    # remove loop over k
    # ret3 = np.zeros((M, K))
    # for n in range(N):
    #     ret3 += np.outer(Z[n], T[n] - Y[n])
    ret4 = np.dot(Z.T,T-Y)
    return ret4

def derivative_b2(T, Y):
    return (T-Y).sum(axis=0)

def derivative_w1(X, Z, T, Y, W2):
    N, D = X.shape
    M, K = W2.shape
    # slow method--direct from derivation
    # ret1 = np.zeros((M, K))
    # for n in range(N):
    #     for m in range(M):
    #         for k in range(K):
    #             for d in range(D):
    #                 ret1[d,m] += (T[n,k] - Y[n,k]) * W2[m,k] * Z[n,m] * (1-Z[n,m]) * X[n,d]
    # vectorized form
    # dZ = np.dot((T-Y),W2.T) * Z * (1-Z)
    dz = np.dot((T-Y),W2.T) * Z * (1 - Z)
    return np.dot(X.T,dz)

def derivative_b1(T, Y, W2, Z):
    return (np.dot((T-Y), W2.T) * Z * (1-Z)).sum(axis=0)

def cost(T, Y):
    tot = T * np.log(Y)
    return tot.sum()

def main():
    Nclass = 500
    D = 2 # number of features
    M = 8 # number of hidden units
    K = 3 # number of classes

    X1 = np.random.randn(Nclass, D) + np.array([0, -2])
    X2 = np.random.randn(Nclass, D) + np.array([2, 2])
    X3 = np.random.randn(Nclass, D) + np.array([-2, -2])
    X = np.vstack([X1, X2, X3])

    Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
    N = len(Y)
    T = np.zeros((N, K))
    for i in range(N):
        T[i, Y[i]] = 1 # one hot encoding for targets

    plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
    plt.show()

    W1 = np.random.randn(D,M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)
    learning_rate = 10e-7
    costs = []
    for epoch in range(100000):
        output, hidden = forward(X, W1, b1, W2, b2)
        if epoch % 100 == 0:
            c = cost(T, output)
            P = np.argmax(output, axis=1)
            r = classification_rate(Y, P)
            print("Cost: {}, Classification Rate: {}".format(c,r))
            costs.append(c)

        W2 += learning_rate * derivative_w2(hidden, T, output)
        b2 += learning_rate * derivative_b2(T, output)
        W1 += learning_rate * derivative_w1(X, hidden, T, output, W2)
        b1 += learning_rate * derivative_b1(T, output, W2, hidden)

    plt.plot(costs)
    plt.show()

if __name__ == '__main__':
    main()
