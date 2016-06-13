import numpy as np
import os
import math
import scipy.sparse as sps
from sklearn.utils import shuffle
import time
import pmf

'''
Helper function of initialization

:return: column vector of weight matrix W
'''
def init(num_feature):
    W = np.random.rand(num_feature, 1)*0.5
    return np.matrix(W)

'''
Helper function of calculating sigmoid function

:return: column vector of sigmoid function sig(z)
'''
def getSigmoid(W, x):
    z = x.dot(W)
    z = np.exp(-1*z)
    z = 1.0 / (1+z)
    # print z
    # z.data[:] = np.exp(-z.data)
    # z.data[:] = 1.0 / (1+z.data)
    return z

'''
Helper function of get training data matrix

:return: data matrix and label matrix
'''
def getTrainData():
    label = []
    data = []
    r = 0
    c = 0
    with open("../data/rank_train.data", 'r') as f:
        for line in f:
            val = line.strip().split(" ")
            c = len(val)-1
            label.append(1) if int(val[0]) == 1 else label.append(0)
            for i in xrange(1, len(val)):
                data.append(float(val[i].split(":")[1]))
            r += 1
            # if r > 1000:
            #     break
    print "number of instance: ", r
    return np.matrix((data)).reshape((r, c)), np.matrix((label)).reshape((r, 1))


'''
Training Process

:return: weight matrix W
'''
def train():
    data, label = getTrainData()
    print "DATA READ!"
    # data = sps.csr_matrix(data)
    # label = sps.csr_matrix(label)
    print "START TRAINING!"
    num_feature = data.shape[1]
    print "NUM OF FEATURES: ", num_feature
    epsilon = 0.05    # learning rate
    lamda = 0.005       # reguarization factor
    num_batch = 800 # 7000
    momentum = 0.4
    maxIter = 1500  # 1500

    W = init(num_feature)
    # W = sps.csr_matrix(W)
    # prev_W_grad = sps.csr_matrix(W.shape)
    prev_W_grad = np.zeros(W.shape)
    prev_train_error = 10.0

    print "start shuffle"
    shuffled_data, shuffled_label = shuffle(data, label)
    time_start = time.time()
    for iter in xrange(maxIter):
        train_error = 0
        batch_size = data.shape[0]/num_batch
        for batch in xrange(num_batch):
            # print "start batch"
            # get batch size
            start = batch*batch_size
            end = (batch+1)*batch_size if batch < num_batch-1 else data.shape[0]
            batch_size = batch_size if batch < num_batch-1 else data.shape[0]-batch*batch_size
            batch_data = shuffled_data[start:end, :]
            batch_label = shuffled_label[start:end, :]
            # print "start calculating sigmoid"
            sigmoid_z = getSigmoid(W, batch_data)
            # calculate gradient
            # print "start calculating gradient"
            delta = batch_label - sigmoid_z
            train_error += np.multiply(delta, delta).sum()
            dW = batch_data.transpose().dot(delta)
            dW -= lamda * W
            # update weights
            # print "start updating weights"
            W_grad = momentum * prev_W_grad + epsilon * dW
            W += W_grad
            prev_W_grad = W_grad
        # epsilon *= 0.995
        train_error = train_error/data.shape[0]
        if math.fabs(train_error-prev_train_error) < 1e-15:
            break
        prev_train_error = train_error
        print "train_error:", train_error, " iter: ", iter
    print "traning finished!"
    duration = time.time() - time_start
    print "running time: ", duration
    return W

def pred(W, isDev):
    # W = W.todense()
    print "START PREDICTING!"
    if os.path.exists("../data/rank_result"):
        os.system("rm -f ../data/rank_result")
    if isDev:
        filepath = "../data/rank_dev.data"
    else:
        filepath = "../data/rank_test.data"
    raw_data = []
    r = 0
    c = 0
    with open(filepath, 'r') as f:
        for line in f:
            val = line.strip().split(" ")
            c = len(val) - 1
            for i in xrange(1, len(val)):
                raw_data.append(float(val[i].split(":")[1]))
            r += 1
            # if r > 1000:
            #  break
    test_data = np.matrix((raw_data)).reshape((r, c))
    predict = test_data.dot(W)
    predict = np.exp(-1*predict)
    predict = 1.0 / (1+predict)
    with open("../data/rank_result", "w") as myfile:
        for i in xrange(predict.shape[0]):
            myfile.write("1" + "\n") if predict[i,0] > 0.5 else myfile.write("-1" + "\n")
    return

def pred_direct(W, isDev):
    U = pmf.loadLatentMat("U")
    V = pmf.loadLatentMat("V")
    if isDev:
        filepath = "../data/dev.csv"
    else:
        filepath = "../data/test.csv"
    if os.path.exists("../data/rank_predictions"):
        os.system("rm -f ../data/rank_predictions")

    rst_lst = []
    with open(filepath, "r") as myfile:
        for line in myfile:
            pair = line.strip().split(",")
            movie_id = int(pair[0])
            user_id = int(pair[1])
            vec = np.multiply(U[user_id, :], V[movie_id, :])
            rst_lst.append(float(vec.dot(W)))
    with open("../data/rank_predictions", "w") as myfile:
        for pred in rst_lst:
            myfile.write(str(pred) + "\n")
    return



if __name__ == "__main__":
    a = np.matrix([[1,1,1,1],[1,2,3,6], [3,2,7,1]])
    print np.sum(a)
    # W = init(4)
    # W2 = normalize(W, norm='l1', axis=0)
    # print W
    # print W2