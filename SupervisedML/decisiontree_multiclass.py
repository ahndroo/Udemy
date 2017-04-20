import numpy as np
from util import get_train_data, get_xor, get_donut
from datetime import datetime

def entropy(y):
    N = len(y)
    classes = np.unique(y)
    numClass=len(classes)
    sums = np.zeros((numClass,1))
    for i,c in enumerate(classes):
        # number of samples per class
        sums[i] = (y==c).sum()
    probs = sums / N #calculate probabilities
    return float(sum(-probs*np.log2(probs)))

class TreeNode:
    def __init__(self, depth=0,max_depth=None):
        self.depth = depth
        self.max_depth = max_depth

    def fit(self, X, Y):
        if len(Y) == 1 or len(set(Y)) == 1: # case of only one label in Y
            self.col = None
            self.split = None
            self.left = None
            self.right = None
            self.prediction = Y[0]
        else:
            D = X.shape[1]
            cols = range(D)
            max_ig = 0
            best_col = None
            best_split = None
            for col in cols:
                ig, split = self.find_split(X, Y, col)
                if ig > max_ig:
                    max_ig = ig
                    best_col = col
                    best_split = split
            # no more information to be gained--predict model outputs
            # base case...
            if max_ig == 0:
                print("Base case reached...predict")
                self.col = None
                self.split = None
                self.left = None
                self.right = None
                self.prediction = np.round(Y.mean())
            else:
                # keep track of best col and split
                self.col = best_col
                self.split = best_split
                # next base case when max depth reached
                if self.depth == self.max_depth:
                    print("Reached max depth...")
                    self.left = None
                    self.right = None
                    self.prediction = [
                        np.round(Y[X[:,best_col] < self.split].mean()),
                        np.round(Y[X[:,best_col] >= self.split].mean()),
                    ]
                else:
                    # not in a base case- do recursion
                    print("Splitting...")
                    left_idx = (X[:, best_col] < best_split)
                    Xleft = X[left_idx]
                    Yleft = Y[left_idx]
                    self.left = TreeNode(self.depth + 1, self.max_depth)
                    self.left.fit(Xleft, Yleft)

                    right_idx = (X[:, best_col] >= best_split)
                    Xright = X[right_idx]
                    Yright = Y[right_idx]
                    self.right = TreeNode(self.depth + 1, self.max_depth)
                    self.right.fit(Xright, Yright)

    def tile_data(self,X,Y,tilesize=49):
        #divide MNIST images into 7x7px tiles
        N,D = X.shape
        numTile = int(D/tilesize)
        xTile = np.zeros((N,numTile))
        for i in range(numTile):
            tiledata = X[:,i*tilesize:(i+1)*tilesize]
            xTile[:,i] = np.mean(tiledata,axis=1) #what do this

    def find_split(self, X, Y, col):
        # sorts data from lowest value to highest
        x_values = X[:,col]
        sort_idx = np.argsort(x_values)
        x_values = x_values[sort_idx]
        y_values = Y[sort_idx]
        labels = np.unique(y_values)
        # numpy nonzero returns elements that are non-zero
        # y_values[:-1] != y_values[1:] compares y_values to one unit shifted
        # version of itself to see if the values match--True indicates boundary
        # boundaries contains non-zero elements in previous comparision
        boundaries = np.nonzero(y_values[:-1] != y_values[1:])[0]
        best_split = None
        max_ig = 0
        # loop over all boundaries until best split is found (with most info gain)
        for b in boundaries:
            split = (x_values[b] + x_values[b+1])/2
            ig = self.information_gain(x_values, y_values, split)
            if ig > max_ig:
                max_ig = ig
                best_split = split
        return max_ig, best_split

    def information_gain(self, x, y, split):
        y0 = y[x < split]
        y1 = y[x >= split]
        N = len(y)
        y0len = len(y0)
        y1len = len(y1)
        if y0len == 0 or y0len == N:
            return 0
        classes0 = np.unique(y0)
        classes1 = np.unique(y1)
        sums0 = np.zeros(len(classes0))
        sums1 = np.zeros(len(classes1))
        for i,c0 in enumerate(classes0):
            sums0[i] = (y0==c0).sum()
        for j,c1 in enumerate(classes1):
            sums1[j] = (y1==c1).sum()
        prob0 = sums0 / N
        prob1 = sums1 / N
        return float(entropy(y) - sum(prob0*entropy(y0)) - sum(prob1*entropy(y1)))

    def predict_one(self, x):
        if self.col is not None and self.split is not None:
            feature = x[self.col]
            if feature < self.split:
                if self.left:
                    p = self.left.predict_one(x)
                else:
                    p = self.prediction[0]
            else:
                if self.right:
                    p = self.right.predict_one(x)
                else:
                    p = self.prediction[1]
        else:
            p = self.prediction
        return p

    def predict(self,X):
        N = len(X)
        P = np.zeros(N)
        for i in range(N):
            P[i] = self.predict_one(X[i])
        print(self.depth)
        return P

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, Y):
        print("Training data...")
        self.root = TreeNode(max_depth = self.max_depth)
        self.root.fit(X, Y)

    def predict(self, X):
        return self.root.predict(X)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

if __name__ == '__main__':
    X, Y = get_train_data()
    N = int(len(Y)/4)
    X = X[:N];Y=Y[:N] # divide data in quarter
    Ntrain = int(len(Y)*.8) #use 80% of data for train, 20% for valid
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xvalid, Yvalid = X[Ntrain:], Y[Ntrain:]

    model = DecisionTree(max_depth=100)
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)

    print("Training time ",datetime.now()-t0)

    t0 = datetime.now()
    print("Train accuracy: ", model.score(Xtrain, Ytrain))
    print("Time to compute train accuracy: ", datetime.now()-t0, "\nTrain size: ", len(Ytrain))

    t0 = datetime.now()
    testScore = model.score(Xvalid,  Yvalid)
    print(" Test accuracy: ", testScore)
    print("Time to compute test accuracy: ", datetime.now()-t0, "\nTest size: ", len(Yvalid), "\n")
