import re
import iohelper
import heapq
import numpy as np
import scipy.sparse as sps
import os
import math
from stemming.porter2 import stem

def preprocess(text, stop_words):
    # text = text.lower()
    # remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    # text = text.translate(remove_punctuation_map)
    # text = ' '.join(word for word in text.split() if not any(c.isdigit() for c in word))
    text = re.sub('[^\w\s]|\w*\d\w*', '', text.lower())
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def toBOW(text):
    dict = {}
    words = text.split()
    for word in words:
        # word = stem(word)
        if not dict.has_key(word):
            dict[word] = 1
        else:
            dict[word] += 1
    return dict

def getCF_Dict(lst_BOW, num_feature):
    dict = {}
    for bow in lst_BOW:
        for word in bow.keys():
            if not dict.has_key(word):
                dict[word] = bow[word]
            else:
                dict[word] += bow[word]
    CFDict = selectTop(dict, num_feature)
    print CFDict
    return CFDict, dict

def getDF_Dict(lst_BOW, num_feature):
    dict = {}
    for bow in lst_BOW:
        for word in bow.keys():
            if not dict.has_key(word):
                dict[word] = 1
            else:
                dict[word] += 1
    DFDict = selectTop(dict, num_feature)
    print DFDict
    return DFDict, dict

# mixture of CF and DF dictionary
def getCFDF_Dict(lst_BOW, num_feature):
    dict_top_CF, dict_CF = getCF_Dict(lst_BOW, num_feature)
    dict_top_DF, dict_DF = getDF_Dict(lst_BOW, num_feature)
    dict_CFDF = dict_top_CF + list(set(dict_top_DF) - set(dict_top_CF))
    return dict_CFDF, dict_CF, dict_DF


def getData(lst_train_BOW, lst_dev_BOW, lst_test_BOW, num_feature, isCF, isDF):
    dict_top = []
    if isCF and isDF:
        dict_top, _, _ = getCFDF_Dict(lst_train_BOW, num_feature)
    elif isCF:
        dict_top, _ = getCF_Dict(lst_train_BOW, num_feature)
    elif isDF:
        dict_top, _ = getDF_Dict(lst_train_BOW, num_feature)

    index = range(len(dict_top))
    dict_index = dict(zip(dict_top, index))
    dict_top = set(dict_top)

    mat_train = getData_helper(lst_train_BOW, dict_top, dict_index, num_feature)
    mat_dev = getData_helper(lst_dev_BOW, dict_top, dict_index, num_feature)
    mat_test = getData_helper(lst_test_BOW, dict_top, dict_index, num_feature)
    return mat_train, mat_dev, mat_test

def getData_custom(lst_train_BOW, lst_dev_BOW, lst_test_BOW, num_feature, isCF, isDF):
    dict_top = []
    if isCF and isDF:
        dict_top, dict_CF, dict_DF = getCFDF_Dict(lst_train_BOW, num_feature)
    elif isCF:
        dict_top, dict_CF = getCF_Dict(lst_train_BOW, num_feature)
        _, dict_DF = getDF_Dict(lst_train_BOW, num_feature)
    elif isDF:
        dict_top, dict_DF = getDF_Dict(lst_train_BOW, num_feature)
        _, dict_CF = getCF_Dict(lst_train_BOW, num_feature)
    else:
        _, dict_CF = getCF_Dict(lst_train_BOW, num_feature)
        _, dict_DF = getDF_Dict(lst_train_BOW, num_feature)

    N = len(lst_train_BOW)
    Ctf = 0
    for word in dict_CF.keys():
        Ctf += dict_CF[word]
    avgLen = float(Ctf)/N

    index = range(len(dict_top))
    dict_index = dict(zip(dict_top, index))
    dict_top = set(dict_top)

    mat_train = getData_custom_helper(lst_train_BOW, dict_top, dict_index, num_feature, dict_DF, avgLen, N)
    mat_dev = getData_custom_helper(lst_dev_BOW, dict_top, dict_index, num_feature, dict_DF, avgLen, N)
    mat_test = getData_custom_helper(lst_test_BOW, dict_top, dict_index, num_feature, dict_DF, avgLen, N)
    return mat_train, mat_dev, mat_test

def getData_helper(lst_BOW, dict_top, dict_index,num_feature):
    num_feature = len(dict_top) if len(dict_top) > 0 else num_feature
    print "number of features: ", num_feature
    tmp_mat_lil = sps.lil_matrix((len(lst_BOW), num_feature))
    row = []
    col = []
    data = []
    for i in xrange(len(lst_BOW)):
        bow = lst_BOW[i]
        for word in bow.keys():
            if len(dict_top) > 0:
                if word in dict_top:
                    row.append(i)
                    data.append(bow[word])
                    col.append(dict_index[word])
            else:
                tmp_mat_lil[i, hash(word)%num_feature] += bow[word]
    if len(dict_top) > 0:
        tmp_mat= sps.coo_matrix((data, (row, col)),shape=(len(lst_BOW), num_feature))
        mat = sps.csr_matrix(tmp_mat)
    else:
        mat = sps.csr_matrix(tmp_mat_lil)
    return mat

def getData_custom_helper(lst_BOW, dict_top, dict_index, num_feature, dict_DF, avgLen, N):
    num_feature = len(dict_top) if len(dict_top) > 0 else num_feature
    print "number of features: ", num_feature
    tmp_mat_lil = sps.lil_matrix((len(lst_BOW), num_feature))
    row = []
    col = []
    data = []
    for i in xrange(len(lst_BOW)):
        bow = lst_BOW[i]
        docLen = 0
        for word in bow.keys():
            docLen += bow[word]
        for word in bow.keys():
            df = 0 if not dict_DF.has_key(word) else dict_DF[word]
            # score = bow[word] * math.log(1 + N/(df+1) ) * 0.6
            score = bow[word] * math.log((N-df+0.5) / (df+0.5)) / (docLen/avgLen)
            if len(dict_top) > 0:
                if word in dict_top:
                    row.append(i)
                    data.append(score)
                    col.append(dict_index[word])
            else:
                tmp_mat_lil[i, hash(word)%num_feature] += score
    if len(dict_top) > 0:
        tmp_mat= sps.coo_matrix((data, (row, col)),shape=(len(lst_BOW), num_feature))
        mat = sps.csr_matrix(tmp_mat)
    else:
        mat = sps.csr_matrix(tmp_mat_lil)
    return mat

# helper function for select top ranked words
def selectTop(dict, size):
    top = heapq.nlargest(size, dict, key = dict.get)
    return top

def save_mat(mat, path):
    os.system("rm -f ../data/" + path + ".npy")
    np.save("../data/" + path, mat)

def load_mat(path):
    mat = np.load("../data/" + path + ".npy")
    return mat

if __name__ == "__main__":
    # print stem("factionally")
    prior = [0.2/0.10199, 0.2/0.08965, 0.2/0.14196, 0.2/0.29750, 0.2/0.36689]
    aaa = np.array(prior)
    aaa = np.matrix(aaa)
    aaa = aaa.transpose()
    bbb = np.repeat(aaa, 10, axis=1)
    print bbb
    # print load_mat("train")
    # getDF_Dict(lst_BOW, 2000)