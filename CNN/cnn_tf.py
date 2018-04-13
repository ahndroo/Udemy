import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from scipy.signal import convolve2d
from scipy.io import loadmat
from sklearn.utils import shuffle
from benchmark import get_data, y2indicator, error_rate

def convpool(X, W, b):
    # assume pool_sz is (2,2) because we need to augment with 1s
    conv_out = tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='SAME')
    conv_out = tf.nn.bias_add(conv_out, b)
        # """ tf.nn.conv2d
        # input: A Tensor. Must be one of the following types: half, float32. A 4-D tensor. The dimension order is interpreted
        #     according to the value of data_format, see below for details.
        # filter: A Tensor. Must have the same type as input. A 4-D tensor of shape [filter_height, filter_width, in_channels,
        #     out_channels]
        # strides: A list of ints. 1-D tensor of length 4. The stride of the sliding window for each dimension of input. The
        #     dimension order is determined by the value of data_format, see below for details.
        #     In this example, stride is simply 1 in all dimensions
        # padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
        # use_cudnn_on_gpu: An optional bool. Defaults to True.
        # data_format: An optional string from: "NHWC", "NCHW". Defaults to "NHWC". Specify the data format of the input and
        #     output data. With the default format "NHWC", the data is stored in the order of: [batch, height, width, channels]. Alternatively, the format could be "NCHW", the data storage order of: [batch, channels, height, width].
        # name: A name for the operation (optional).
        # """
    pool_out = tf.nn.max_pool(conv_out, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        # """ tf.nn.max_pool
        # value: A 4-D Tensor with shape [batch, height, width, channels] and type tf.float32.
        # ksize: A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
        #     Ex. ksize=[1,2,2,1] creates window of size 1 in N and color channel, 2x2 in input W and H
        # strides: A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.
        #     Ex. stride=[1,2,2,1] will traverse N and color channel with step_sz 1, and 4-D tensor w/ step_sz 2
        # padding: A string, either 'VALID' or 'SAME'. The padding algorithm. See the comment here
        # data_format: A string. 'NHWC' and 'NCHW' are supported.
        # name: Optional name for the operation.
        # """
    return tf.nn.relu(pool_out)

def init_filter(shape, poolsz):
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(poolsz)))
    return w.astype(np.float32)

def rearrange(X):
    # .mat input has shape (32,32,3,N)
    # tensorflow expects (N, 32, 32, 3)
    return(X.transpose(3,0,1,2) / 255).astype(np.float32) # scale data (in: 0:255, out: 0:1)

def main():
    train, test = get_data()
    Xtrain = rearrange(train['X'])
    # train['y'] has shape (N,1) and vals ranging 1:10; need shape (N,) and ranging 0:9
    Ytrain = train['y'].flatten() - 1
    del train
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
    Ytrain_ind = y2indicator(Ytrain)

    Xtest = rearrange(test['X'])
    Ytest = test['y'].flatten() - 1
    del test
    Ytest_ind = y2indicator(Ytest)

    # grad. desc. params
    max_iter = 20
    print_period = 10
    N = Xtrain.shape[0]
    batch_sz = 500
    n_batches = N // batch_sz

    # make num samples divisible by batch_sz so all batches are same sz
    Xtrain = Xtrain[:73000,]
    Ytrain = Ytrain[:73000]
    Xtest = Xtest[:26000,]
    Ytest = Ytest[:26000]
    Ytest_ind = Ytest_ind[:26000,]

    # initial weights
    M = 500 # neurons in final layer
    K = 10 # num classes
    pool_sz = (2,2)

    W1_shape = (5, 5, 3, 20) # filter shape (width, height, num_color_channel, num_feature_maps(or filters))
    W1_init = init_filter(W1_shape, pool_sz) # pass in pool_sz for normalization
    b1_init = np.zeros(W1_shape[-1], dtype=np.float32)

    W2_shape = (5,5,20,50)
    W2_init = init_filter(W2_shape, pool_sz)
    b2_init = np.zeros(W2_shape[-1], dtype=np.float32)

    # vanilla NN weights
    W3_init = np.random.randn(W2_shape[-1]*8*8, M) / np.sqrt(W2_shape[-1]*8*8 + M) # 8 factor is result of
        # final convolution (2 convpool layers 32x32--> 16x16 --> 8x8 output_sz)
    b3_init = np.zeros(M, dtype=np.float32)
    W4_init = np.random.randn(M, K) / np.sqrt(M + K)
    b4_init = np.zeros(K, dtype=np.float32)

    X = tf.placeholder(tf.float32, shape=(batch_sz, 32, 32, 3), name='X')
    T = tf.placeholder(tf.float32, shape=(batch_sz, K), name='T')
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))
    W4 = tf.Variable(W4_init.astype(np.float32))
    b4 = tf.Variable(b4_init.astype(np.float32))

    Z1 = convpool(X, W1, b1)
    Z2 = convpool(Z1, W2, b2)
    Z2_shape = Z2.get_shape().as_list()
    Z2r = tf.reshape(Z2, [Z2_shape[0], np.prod(Z2_shape[1:])]) # reshape data to input to ANN layer
    Z3 = tf.nn.relu(tf.matmul(Z2r, W3) + b3)
    Yish = tf.matmul(Z3, W4) + b4

    cost = tf.reduce_sum( # sums all elements in matrix
        tf.nn.softmax_cross_entropy_with_logits(
            # computes softmax with logits and labels
            logits= Yish,
            labels=T
        )
    )

    train_op = tf.train.RMSPropOptimizer(0.0001, decay=0.99, momentum=0.9).minimize(cost)

    predict_op = tf.argmax(Yish ,1)

    t0 = datetime.now()
    LL = []

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i in range(max_iter):
            for j in range(n_batches):
                Xbatch = Xtrain[j*batch_sz:(j*batch_sz+batch_sz),]
                Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz+batch_sz)]

                if len(Xbatch) == batch_sz:
                    sess.run(train_op, feed_dict={X: Xbatch, T: Ybatch})
                    if j % print_period == 0:
                        # due to RAM limiations, we need to have fixed size input
                        # as a result, need total clost and pred computation
                        test_cost = 0
                        prediction = np.zeros(len(Xtest))
                        # since tf var X is expecting input of batch_sz, need to loop throug Xtest
                        # in iterations of batch_sz
                        for k in range(len(Xtest) // batch_sz):
                            Xtestbatch = Xtest[k*batch_sz:(k*batch_sz+batch_sz),]
                            Ytestbatch = Ytest_ind[k*batch_sz:(k*batch_sz + batch_sz)]
                            test_cost += sess.run(cost, feed_dict={X:Xtestbatch, T:Ytestbatch})
                            prediction[k*batch_sz:(k*batch_sz+batch_sz)] = sess.run(
                                predict_op, feed_dict={X:Xtestbatch})
                        err = error_rate(prediction, Ytest)
                        print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
                        LL.append(test_cost)
        print("Elapsed time:", (datetime.now() - t0))
        plt.plot(LL)
        plt.show()

if __name__ == '__main__':
    main()
