import numpy as np
import matplotlib.pyplot as plt
from knn import KNN

def get_data():
    h = 8
    w = 8
    N = h * w
    X = np.zeros((N, 2))
    Y = np.zeros(N)
    n = 0
    start_t = 0
    for i in range(w):
        t = start_t
        for j in range(h):
            # X array contains all data points
            X[n] = [i, j]
            Y[n] = t
            n += 1
            t = (t+1) % 2
        start_t = (start_t + 1) % 2
    return X, Y

if __name__ == '__main__':
    X, Y = get_data()
    plt.scatter(X[:,0],X[:,1], s = 100, c=Y, alpha = .5)
    plt.show()

    model = KNN(3)
    model.fit(X, Y)
    print("The accuracy is: ", model.score(X, Y))
