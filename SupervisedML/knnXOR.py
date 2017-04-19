<<<<<<< HEAD
from util import get_xor
from knn import KNN
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X, Y = get_xor()

    plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=.5)
    plt.show()

    model = KNN(3)
    model.fit(X, Y)
    print("Accuracy: ", model.score(X,Y))
=======
from util import get_xor
from knn import KNN
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X, Y = get_xor()

    plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=.5)
    plt.show()

    model = KNN(3)
    model.fit(X, Y)
    print("Accuracy: ", model.score(X,Y))
>>>>>>> 6b00251a86bc52850aa5b4d784341059d3aa9b4e
