
import matplotlib.pyplot as plt
import numpy as np
from theano_ann import ANN
from util import get_spiral, get_clouds
from sklearn.utils import shuffle

def random_search():
    X, Y = get_spiral()
    # X, Y = get_clouds()
    X, Y = shuffle(X, Y)
    Ntrain = int(0.7*len(X))
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    # starting parameters
    M = 20
    nHidden = 2
    log_lr = -4 # use log values, since typical adjustments are .1, .01...., .000001 etc
    log_l2 = -2
    max_tries = 30

    # loop through all possible hyperparameter settings
    best_validation_rate = 0
    best_hls = None
    best_lr = None
    best_l2 = None
    for _ in range(max_tries):
        model = ANN([M]*nHidden)
        # raise 10 to log_lr and log_l2 power since they're in log scale
        model.fit(Xtrain, Ytrain, learning_rate=10**log_lr, reg=10**log_l2, mu=0.99, epochs=3000, show_fig=False)
        validation_accuracy = model.score(Xtest, Ytest)
        train_accuracy = model.score(Xtrain, Ytrain)
        print("validation_accuracy: %.3f, train_accuracy: %.3f, settings: %s, %s, %s" %
            (validation_accuracy, train_accuracy, [M]*nHidden], log_lr, log_l2)
            )
        if validation_accuracy > best_validation_rate:
            best_validation_rate = validation_accuracy
            best_M = M
            best_nHidden = nHidden
            best_lr = log_lr
            best_l2 = log_l2

        # Select new hyperparameters
        nHidden = best_nHidden * np.random.randint(-1,2) # -1, 0, or 1
        nHidden = max(1,nHidden)
        M = best_M + np.random.randint(-1,2)*10
        M = max(10,M)
        log_lr = best_lr + np.random.randint(-1,2)
        log_l2 = best_l2 + np.random.randint(-1,2)

    print("Best validation_accuracy:", best_validation_rate)
    print("Best Settings:")
    print("hidden_layer_sizes:", best_hls)
    print("learning_rate:", best_lr)
    print("l2:", best_l2)
