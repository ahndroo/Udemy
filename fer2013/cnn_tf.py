import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from datetime import datetime
from sklearn.utils import shuffle
from util import getImageData, init_weight_and_bias, init_filter,  error_rate, y2indicator, makeDiv

class Convpool(object):
    def __init__(self, inputMap_sz, outputMap_sz, filter_W, filter_H, pool_sz=(2,2)):
        filter_sz = (filter_W, filter_H, inputMap_sz, outputMap_sz)
        W0 = init_filter(filter_sz, pool_sz)
        self.W = tf.Variable(W0)
        b0 = np.zeros(outputMap_sz, dtype=np.float32)
        self.b = tf.Variable(b0)
        self.pool_sz = pool_sz
        self.params=[self.W, self.b]

    def forward(self, X):
        conv_out = tf.nn.conv2d(X, self.W, strides=[1,1,1,1], padding='SAME')
        conv_out = tf.nn.bias_add(conv_out, self.b)
        pool_out = tf.nn.max_pool(conv_out, ksize=[1,2,2,1], strides=[1,self.pool_sz[0], self.pool_sz[1],1], padding='SAME')
        return tf.nn.relu(pool_out) # PLAY WITH DIFFERENT ACT FUNCTIONS N STUFF---CREATE FUNC TO CHOOSE DIFFERENT ONES

class HiddenLayer(object):
    def __init__(self, input_sz, output_sz):
        W0 = np.random.randn(input_sz, output_sz) / np.sqrt(input_sz + output_sz)
        W0 = W0.astype(np.float32)
        self.W = tf.Variable(W0)
        b0 = np.zeros(output_sz, dtype=np.float32)
        self.b = tf.Variable(b0)
        self.params = [self.W, self.b]

    def forward(self, X):
        return tf.nn.relu(tf.matmul(X, self.W) + self.b)

class CNN(object):
    """
    CNN Architecture using Tensorflow framework
        - General purpose CNN.  Input data expected to be of shape (N, im_W, im_H, RGB)
        - more shit
    """

    def __init__(self, convpool_sz, hidden_sz):
        # convpool_sz is a list with each entry of shape (num_feature_maps, filter_W, filter_H)
        # hidden_sz is list where each entry is num neurons in MLP layer
        self.convpool_sz = convpool_sz
        self.hidden_sz = hidden_sz

    def train(self, X, Y, learning_rate=10e-4, mu=.99, reg=10e-4, decay=0.9999, eps=10e-3, batch_sz=100, epochs=3, dispFig=True):
        print('Training model...')
        # tensorflow expects inputs to be of same data format
        learning_rate = np.float32(learning_rate)
        mu = np.float32(mu)
        decay = np.float32(decay)
        eps = np.float32(eps)

        # input data should have shape (N, im_W, im_H, color_channels)
        X, Y = shuffle(X, Y)
        N, im_W, im_H, color_channels = X.shape
        K = len(np.unique(Y)) # number of classes
        # check if input truths are vector or one hot encoded
        if len(Y.shape)==1  or Y.shape[1]!=K:
            Y_ind = y2indicator(Y).astype(np.float32)
        else:
            Y_ind = Y
        X = X.astype(np.float32) # just a precaution...

        # use 80% of data for test, 20% for validation set
        # initialize tensorflow var X with shape (NONE, w,h,color)
        numTrain = round(N * .8)
        numTest = round(N * .2)
        trainIdx = makeDiv(numTrain, batch_sz)
        validIdx = makeDiv(numTest, batch_sz)
        Xtrain = X[:trainIdx,]; Ytrain = Y_ind[:trainIdx,]
        Xvalid = X[-validIdx:,]; Yvalid = Y_ind[-validIdx:,]

        # init Convpool layers
        inputMap_sz = X.shape[-1]
        self.convpoolLayers = []
        outW = im_W; outH = im_H
        for outMap, filter_W, filter_H in self.convpool_sz:
            self.convpoolLayers.append(Convpool(inputMap_sz, outMap, filter_W, filter_H))
            inputMap_sz = outMap
            outW = outW // 2
            outH = outH // 2

        # init MLP layers
        self.hiddenLayers = []
        hiddenInput_shp = inputMap_sz*outW*outH
        for m in self.hidden_sz:
            self.hiddenLayers.append(HiddenLayer(hiddenInput_shp, m))
            hiddenInput_shp = m
        V, c = init_weight_and_bias(hiddenInput_shp, K)
        self.V = tf.Variable(V)
        self.c = tf.Variable(c)

        # collect params for use in updates
        self.params = [self.V, self.c]
        for h in self.convpoolLayers:
            self.params += h.params
        for h in self.hiddenLayers:
            self.params += h.params

        tfX = tf.placeholder(tf.float32, shape=(None, im_W, im_H, color_channels), name='X')
        tfY = tf.placeholder(tf.float32, shape=(None, K), name='Y')
        Z_logreg = self.forward(tfX)

        rcost = reg*sum([tf.nn.l2_loss(p) for p in self.params]) # calculate l2 penalty
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=Z_logreg,
                labels = tfY
            )
        ) + rcost
        prediction = self.predict(tfX)

        train_op = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=mu).minimize(cost)

        n_batches = len(Xtrain) // batch_sz

        costs = []
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(epochs):
                Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
                for j in range(n_batches):
                    Xbatch = Xtrain[j*batch_sz:(j*batch_sz+batch_sz),]
                    Ybatch = Ytrain[j*batch_sz:(j*batch_sz+batch_sz),]

                    sess.run(train_op, feed_dict={tfX:Xbatch, tfY:Ybatch})

                    if j % 10 == 0:
                        c = sess.run(cost, feed_dict={tfX:Xvalid, tfY:Yvalid})
                        costs.append(c)

                        p = sess.run(prediction, feed_dict={tfX:Xvalid, tfY:Yvalid})
                        e = error_rate(np.argmax(Yvalid,axis=1), p)
                        print('Epoch: {}\t batch: {}\t cost: {}\t error: {}'.format(i, j, c, e))

        print('Final Accuracy: {}'.format(1-e))
        if dispFig:
            plt.plot(costs)
            plt.xlabel('Epochs'); plt.ylabel('Cost')
            plt.show()
        return costs, (1-e)

    def forward(self, X):
        Z = X
        for c in self.convpoolLayers:
            Z = c.forward(Z)
        Z_shape = Z.get_shape().as_list()
        Z = tf.reshape(Z, [-1, np.prod(Z_shape[1:])]) # -1 arg keeps that dim constant in reshaping
        for h in self.hiddenLayers:
            Z = h.forward(Z)
        return tf.matmul(Z, self.V) + self.c

    def predict(self, X):
        pY = self.forward(X)
        return tf.argmax(pY, 1)

def main():
    X, Y = getImageData()
    X = X.transpose(0,2,3,1).astype(np.float32)

    model = CNN(convpool_sz=[(20, 5, 5), (20, 5, 5)],
                hidden_sz=[500, 300])
    t0 = datetime.now()
    costs, acc = model.train(X, Y, epochs=500, batch_sz=100)
    print('Time to train for 1000 epochs w/ batch sz 500: {}'.format(datetime.now()-t0))

if __name__ == '__main__':
    main()
