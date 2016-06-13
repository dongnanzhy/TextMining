import numpy as np
import os
import math
import eval
import scipy.sparse as sps
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
import iohelper
import preprocess

# each row is corresponded to one weight vector of label c
def init(num_feature):
    W = np.random.rand(5, num_feature)*0.05
    return np.matrix(W)

# each row is probability distribution of one instance by different label c
def getProb(W, x):
    pred_vec = W.dot(x.transpose())
    pred_vec.data[:] = np.exp(pred_vec.data)
    # pred_vec = np.zeros((5, x.shape[0]))
    # for i in xrange(5):
    #     pred_vec[i, :] = np.exp(tmp[i, :])
    # pred_vec = W.dot(x.transpose()).expm1()
    # pred_vec = np.exp(x.dot(W.transpose()))
    # pred_vec = pred_vec.transpose()
    prob = normalize(pred_vec, norm='l1', axis=0)
    return prob.transpose()

def getError(label, prob):
    prob.data[:] = np.log(prob.data)
    error_mat = label.multiply(prob)
    error = error_mat.sum()
    return error

def train(data, lst_label):
    print "START TRAINING!"
    label = getLabel(lst_label)
    label = sps.csr_matrix(label)
    num_feature = data.shape[1]
    epsilon = 0.0005    # learning rate
    lamda = 0.001       # reguarization factor
    num_batch = 2500
    momentum = 0.4
    maxIter = 1500

    W = init(num_feature)
    W = sps.csr_matrix(W)
    prev_W_grad = sps.csr_matrix(W.shape)
    prev_train_error = 10.0

    prior = [0.2/0.10199, 0.2/0.08965, 0.2/0.14196, 0.2/0.29750, 0.2/0.36689]
    prior_m = np.array(prior)
    prior_m = np.matrix(prior_m)
    prior_m = prior_m.transpose()
    prior_m = np.repeat(prior_m, num_feature, axis=1)
    prior_m = sps.csr_matrix(prior_m)

    for iter in xrange(maxIter):
        shuffled_data, shuffled_label = shuffle(data, label)
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
            # print "start calculating prob"
            prob = getProb(W, batch_data)
            # calculate gradient
            # print "start calculating gradient"
            delta = batch_label - prob
            # train_error += delta.multiply(delta).sum()
            train_error += getError(batch_label,prob)
            dW = delta.transpose().dot(batch_data)
            # dW -= lamda * W.multiply(prior_m)
            dW -= lamda * W
            # update weights
            # print "start updating weights"
            W_grad = momentum * prev_W_grad + epsilon * dW.multiply(prior_m)
            W += W_grad
            prev_W_grad = W_grad
        epsilon *= 0.995
        train_error = train_error/data.shape[0]
        if math.fabs(train_error-prev_train_error) < 1e-8:
            break
        prev_train_error = train_error
        print "train_error:", train_error, " iter: ", iter

    print "traning finished!"
    prob = getProb(W, data)
    # hard predict
    lst_pred_hard = pred_hard_helper(prob)
    # soft predict
    lst_pred_soft = pred_soft_helper(prob)
    print "accuracy:", eval.accuracy(lst_pred_hard, lst_label), " rmse:", eval.rmse(lst_pred_soft, lst_label)
    # save_model(W, "cf_model")
    return W

def pred(data, W):
    prob = getProb(W, data)
    return pred_hard_helper(prob), pred_soft_helper(prob)


def pred_hard_helper(prob):
    prob = prob.todense()
    star = np.argmax(prob, axis=1)
    star += 1
    lst_stars = []
    for row in xrange(star.shape[0]):
        lst_stars.append(star[row, 0])
    # print lst_stars
    return lst_stars

def pred_soft_helper(prob):
    prob = prob.todense()
    r = np.matrix(range(1,6))
    lst_stars = []
    for row in xrange(prob.shape[0]):
        lst_stars.append(np.sum(np.multiply(r, prob[row,:])))
    # print lst_stars
    return lst_stars

def getLabel(lst_stars):
    one_hot_data = np.zeros((len(lst_stars), 5))
    for i in xrange(len(lst_stars)):
        one_hot_data[i, lst_stars[i]-1] = 1
    return one_hot_data

def save_model(W, path):
    os.system("rm -f ../model/" + path + ".npy")
    np.save("../model/" + path, W)

def load_model(path):
    W = np.load("../model/" + path + ".npy")
    return W


if __name__ == "__main__":
    a = np.matrix([[1,1,1,1],[1,2,3,6], [3,2,7,1]])
    print np.sum(a)
    # W = init(4)
    # W2 = normalize(W, norm='l1', axis=0)
    # print W
    # print W2