import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import normalize,scale
import time

NUM_USER = 10916
NUM_MOVIE = 5392


def readTrain():
    tmp = sps.lil_matrix((NUM_USER, NUM_MOVIE))
    with open("../data/train.csv", "r") as myfile:
        for line in myfile:
            val = line.strip().split(",")
            tmp[int(val[1]), int(val[0])] = float(val[2])-3.0
    train = sps.csr_matrix(tmp)
    return train


def model_based(parameter):
    '''
    K: number of neighbors
    metric: cosine or dot-product
    method: mean or weighted-mean
    '''
    K = int(parameter.get("K"))
    metric = parameter.get("metric")
    method = parameter.get("method")
    pcc = True if parameter.get("pcc") == "true" else False
    neibor_explored = {}
    train = readTrain()
    dev_vec=[p.rstrip() for p in open("../data/dev.csv","r").readlines()]
    test_vec=[p.rstrip() for p in open("../data/test.csv","r").readlines()]
    print "data read!"
    start = time.time()
    train_norm = train.copy()
    if pcc:
        print "Model_based PCC Not supported Now!"
    if metric == "cosine":
        train_norm = normalize(train_norm, axis=0)   # normalize by column
    A = train_norm.transpose().dot(train_norm)
    print "Model A calculated!"
    [neibor_explored, dev_pred] = model_helper(K, method, neibor_explored, A, train, dev_vec)
    [neibor_explored, test_pred] = model_helper(K, method, neibor_explored, A, train, test_vec)
    # print "KNN for movie 3: ", neibor_explored[3][0]
    duration = time.time() - start
    print "running time:", duration
    with open("../data/dev_modelCF_pred.csv", "a") as myfile:
        for rate in dev_pred:
            myfile.write(str(rate) + "\n")
    with open("../data/test_modelCF_pred.csv", "a") as myfile:
        for rate in test_pred:
            myfile.write(str(rate) + "\n")

def model_helper(K, method, neibor_explored, A, train, vec):
    pred = []
    for pair in vec:
        val = pair.split(",")
        user = int(val[1])
        movie = int(val[0])
        if not neibor_explored.has_key(movie):
            sim = A[movie,:].copy()
            sim[0,movie] = 0
            sim = sim.todense()
            knn_movie = np.argpartition(-sim, K)[0,:K+1]
            knn_weight = sim[0, knn_movie]
            neibor_explored[movie] = (knn_movie, knn_weight)
        (knn_movie, knn_weight) = neibor_explored.get(movie)
        rate = 0
        if method == "mean":
            for i in xrange(K):
                rate += train[user, knn_movie[0,i]]
            rate /= K
        else:
            weight_norm = normalize(knn_weight,norm='l1')
            for i in xrange(K):
                rate += weight_norm[0,i] * train[user, knn_movie[0,i]]
        rate += 3
        pred.append(rate)
    return neibor_explored, pred


def memory_based(parameter):
    '''
    K: number of neighbors
    metric: cosine or dot-product
    method: mean or weighted-mean
    '''
    K = int(parameter.get("K"))
    metric = parameter.get("metric")
    method = parameter.get("method")
    pcc = True if parameter.get("pcc") == "true" else False
    neibor_explored = {}
    train = readTrain()
    dev_vec=[p.rstrip() for p in open("../data/dev.csv","r").readlines()]
    test_vec=[p.rstrip() for p in open("../data/test.csv","r").readlines()]
    print "data read!"
    start = time.time()
    train_norm = train.copy()
    avgs = np.zeros(train.shape[0])
    stds = np.zeros(train.shape[0])
    if pcc:
        train_norm, avgs, stds = pccUser(train_norm)
    if metric == "cosine":
        train_norm = normalize(train_norm)   # normalize by row
    print "data prepared!"
    [neibor_explored, dev_pred] = memory_helper(K, method, pcc, neibor_explored, train, train_norm, avgs, stds, dev_vec)
    [neibor_explored, test_pred] = memory_helper(K, method, pcc, neibor_explored, train, train_norm, avgs, stds, test_vec)
    duration = time.time()-start
    print "running time:", duration
    # print "KNN for user 4321: ", neibor_explored[4321][0]
    with open("../data/dev_memoryCF_pred.csv", "a") as myfile:
        for rate in dev_pred:
            myfile.write(str(rate) + "\n")
    with open("../data/test_memoryCF_pred.csv", "a") as myfile:
        for rate in test_pred:
            myfile.write(str(rate) + "\n")

def memory_helper(K, method, pcc, neibor_explored, train, train_norm, avgs, stds, vec):
    pred = []
    for pair in vec:
        val = pair.split(",")
        user = int(val[1])
        movie = int(val[0])
        if not neibor_explored.has_key(user):
            sim = train_norm.dot(train_norm[user,:].T)
            sim[user,0] = 0
            sim = sim.transpose()
            sim = sim.todense()
            knn_user = np.argpartition(-sim, K)[0,:K+1]
            knn_weight = sim[0, knn_user]
            neibor_explored[user] = (knn_user, knn_weight)
        (knn_user, knn_weight) = neibor_explored.get(user)
        rate = 0
        std = 1
        avg = 0
        if pcc:
            if avgs.shape[0] == train.shape[0]:
                std = stds[user] if stds[user] > 0 else 1
                avg = avgs[user]
            else:
                std = stds[movie] if stds[movie] > 0 else 1
                avg = avgs[movie]
        if method == "mean":
            for i in xrange(K):
                rate += (train[knn_user[0,i], movie] - avg) / std
            rate /= K
        else:
            weight_norm = normalize(knn_weight,norm='l1')
            for i in xrange(K):
                rate += weight_norm[0,i] * (train[knn_user[0,i], movie] - avg) / std
        rate = rate * std + avg
        rate += 3
        pred.append(rate)
        # print rate
    return neibor_explored, pred
# user bias reduction
def pccUser(train):
    avgs = np.zeros(train.shape[0])
    stds = np.zeros(train.shape[0])
    for row in xrange(train.shape[0]):
        if train[row].size:
            avg = np.mean(train[row,:].data)
            std = np.std(train[row,:].data)
            avgs[row] = avg
            stds[row] = std
            train.data[train.indptr[row]:train.indptr[row+1]] -= avg
            if std > 0:
                train.data[train.indptr[row]:train.indptr[row+1]] /= std
    return train, avgs, stds
# movie bias reduction
def pccMovie(train):
    train = train.transpose()
    avgs = np.zeros(train.shape[0])
    stds = np.zeros(train.shape[0])
    for row in xrange(train.shape[0]):
        if train[row].size:
            avg = np.mean(train[row,:].data)
            std = np.std(train[row,:].data)
            avgs[row] = avg
            stds[row] = std
            train.data[train.indptr[row]:train.indptr[row+1]] -= avg
            if std > 0:
                train.data[train.indptr[row]:train.indptr[row+1]] /= std
    return train.transpose(), avgs, stds

# if __name__ == "__main__":
    # model_based()