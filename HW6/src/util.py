import numpy as np
import scipy.sparse as sps
import os
import math
import collections

NUM_USER = 10916
NUM_MOVIE = 5392

'''
Helper function of generating train/test data

:return: list of movie,rating pair hashed by users
'''
def readTestList(isDev):
    if isDev:
        filepath = "../data/dev.csv"
    else:
        filepath = "../data/test.csv"
    user_dict = {}
    with open(filepath, "r") as myfile:
        for line in myfile:
            pair = line.strip().split(",")
            movie_id = int(pair[0])
            user_id = int(pair[1])
            if user_id in user_dict:
                user_dict[user_id].append(movie_id)
            else:
                user_dict[user_id] = [movie_id]
    return user_dict

def readTrainList():
    filepath = "../data/train.csv"
    user_dict = {}
    with open(filepath, "r") as myfile:
        for line in myfile:
            pair = line.strip().split(",")
            movie_id = int(pair[0])
            user_id = int(pair[1])
            rate = int(pair[2])
            if user_id in user_dict:
                user_dict[user_id].append((movie_id, rate))
            else:
                user_dict[user_id] = [(movie_id, rate)]
    return user_dict


'''
Helper function of write train/test data

:return:
'''
def writeFile(path, label, v):
    with open(path, "a") as myfile:
        s = ""
        s += str(label)
        s += " "
        for col in xrange(v.shape[1]):
            s += str(col+1)
            s += ":"
            if path.__contains__("train"):
                s += "{:0.6f}".format(v[0, col])
            else:
                s += "{:0.8f}".format(v[0, col])
            s += " "
        s += "\n"
        myfile.write(s)
    return

def writePair(lst):
    with open("../data/lst_pair.data", "w") as myfile:
        for pair in lst:
            myfile.write(pair + "\n")
    return


'''
Helper function of rerank

:return: list of movie pair hashed by users
'''
def readResult():
    labels = [int(label.rstrip()) for label in open("../data/rank_result", "r").readlines()]
    pairs = [pair.rstrip() for pair in open("../data/lst_pair.data", "r").readlines()]
    user_dict = {}
    for k in xrange(len(labels)):
        pair = pairs[k].split("-")
        user = int(pair[0])
        movie_i = int(pair[1])
        movie_j = int(pair[2])
        relation = (movie_i, movie_j) if labels[k] > 0 else (movie_j, movie_i)
        if user in user_dict:
            user_dict[user].append(relation)
        else:
            user_dict[user] = [relation]
    return user_dict

'''
Helper function of topological sort

:return: movie rating by sorted order
'''
def topo_sort(relations):
    relations = set(relations)
    dict_movie = {}
    for movie_i, movie_j in relations:
        if movie_i not in dict_movie:
            dict_movie[movie_i] = -1
        if movie_j not in dict_movie:
            dict_movie[movie_j] = -1
    graph = collections.defaultdict(set)
    neighbors = collections.defaultdict(set)
    for prev, follow in relations:
        graph[follow].add(prev)
        neighbors[prev].add(follow)
    stack = [movie for movie in dict_movie.keys() if not graph[movie]]
    result = []
    while stack:
        node = stack.pop()
        result.append(node)
        for n in neighbors[node]:
            graph[n].remove(node)
            if not graph[n]:
                stack.append(n)
    # if len(result) != len(dict_movie):
    #     print "cycle detected!"
    for i in xrange(len(result)):
        dict_movie[result[i]] = len(result) - i
    return dict_movie



'''
Generating train/test data for SVM and LR

:return: list of movie pair hashed by users
'''
def genTrainData(U, V):
    '''
    U: N*D latent user matrix
    V: M*D latent movie matrix
    '''
    if os.path.exists("../data/rank_train.data"):
        os.system("rm -f ../data/rank_train.data")
    filepath = "../data/rank_train.data"
    user_dict = readTrainList()
    print "START GENERATING DATA!"
    for user_id in user_dict.keys():
        v_user = U[user_id, :]
        lst_movie = user_dict[user_id]
        vec_dict = {}
        lst_movie_rate1 = []
        lst_movie_rate5 = []
        for item in lst_movie:
            movie_id = item[0]
            rate = item[1]
            if rate == 1 or rate == 5:
                vec_dict[movie_id] = np.multiply(v_user, V[movie_id, :])
                if rate == 1:
                    lst_movie_rate1.append(movie_id)
                else:
                    lst_movie_rate5.append(movie_id)
        for movie_i in lst_movie_rate5:
            for movie_j in lst_movie_rate1:
                v = vec_dict[movie_i] - vec_dict[movie_j]
                writeFile(filepath, 1, v)
                writeFile(filepath, -1, -v)
        print "user " + str(user_id) + " finished"
        # if user_id > 2:
        #     break
    return

def genTestData(U, V, isDev):
    '''
    U: N*D latent user matrix
    V: M*D latent movie matrix
    '''
    if isDev:
        filepath = "../data/rank_dev.data"
    else:
        filepath = "../data/rank_test.data"
    if os.path.exists(filepath):
        os.system("rm -f " + filepath)
    user_dict = readTestList(isDev)
    print "START GENERATING DATA!"
    lst_pair = []
    for user_id in user_dict.keys():
        v_user = U[user_id, :]
        lst_movie = user_dict[user_id]
        vec_dict = {}
        for movie_id in lst_movie:
            vec_dict[movie_id] = np.multiply(v_user, V[movie_id, :])
        for movie_i in lst_movie:
            for movie_j in lst_movie:
                if movie_i == movie_j:
                    continue
                v = vec_dict[movie_i] - vec_dict[movie_j]
                label = 0
                lst_pair.append(str(user_id) + "-" + str(movie_i) + "-" + str(movie_j))
                writeFile(filepath, label, v)
        print "user " + str(user_id) + " finished"
        # if user_id > 2:
        #     break
    writePair(lst_pair)
    return


'''
Generating final prediction

'''
def genResult(isDev):
    user_dict = readResult()
    predicts = {}
    for user in user_dict:
        # print "generating order for user: ", user
        predicts[user] = topo_sort(user_dict[user])
    if isDev:
        filepath = "../data/dev.csv"
    else:
        filepath = "../data/test.csv"
    lst_result = []
    with open(filepath, "r") as myfile:
        for line in myfile:
            pair = line.strip().split(",")
            movie_id = int(pair[0])
            user_id = int(pair[1])
            if user_id in predicts:
                lst_result.append(predicts[user_id][movie_id])
            else:
                lst_result.append(5)
    with open("../data/rank_predictions", "w") as myfile:
        for rate in lst_result:
            myfile.write(str(rate) + "\n")
    return


if __name__ == "__main__":
    lst = [[0, 1], [0, 2], [1, 3], [2, 3]]
    print topo_sort(lst)