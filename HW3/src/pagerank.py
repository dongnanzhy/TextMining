import scipy.sparse as sps
import numpy as np
import sklearn.preprocessing as skl
import os.path
import math
import time

NUM_DOCS = 81433
NUM_TOPICS = 12
INDRI_LIST= [(2,1), (2,3), (2,4), (5,1), (5,2), (5,3), (5,4), (6,1), (6,2), (6,3), (6,4), (6,5), (12,1), (12,2),
              (12,5), (14,1), (14,2), (14,3), (14,4), (14,5), (15,1), (15,2), (15,3), (15,3),(15,5), (15,6),
              (17,1), (17,2), (17,3), (17,4), (19,1), (19,2), (19,3), (19,5), (20,2), (20,3), (20,4), (20,5)]
start_time = 0

def readParam():
    parameter = {}
    with open("parameter.txt", "r") as myfile:
        for line in myfile:
            val = line.strip().split(":")
            parameter[val[0]] = val[1]
    return parameter
def readTrans():
    tmp = sps.lil_matrix((NUM_DOCS, NUM_DOCS))
    tmp.setdiag(1)
    with open("../data/transition.txt", "r") as myfile:
        for line in myfile:
            val = line.strip().split(" ")
            tmp[int(val[0])-1, int(val[1])-1] = float(val[2])
    trans = sps.csr_matrix(tmp)
    trans_norm = skl.normalize(trans, norm='l1')
    return trans_norm
def readTopic():
    dict = {}
    with open("../data/doc_topics.txt", "r") as myfile:
        for line in myfile:
            val = line.strip().split(" ")
            docID = int(val[0]) - 1
            topicID = int(val[1])
            if dict.has_key(topicID):
                dict.get(topicID).append(docID)
            else:
                dict[topicID] = [docID]
    for topic in dict:
        p = sps.lil_matrix((NUM_DOCS, 1))
        for doc in dict.get(topic):
            p[doc] = 1
        p = sps.csr_matrix(p)
        p = p / p.sum()
        dict[topic] = p
    return dict
def readPrior(isUserPrior):
    dict = {}
    if isUserPrior:
        path = "../data/user-topic-distro.txt"
    else:
        path = "../data/query-topic-distro.txt"
    with open(path, "r") as myfile:
        for line in myfile:
            val = line.strip().split(" ")
            userID = int(val[0])
            queryID = int(val[1])
            pr = np.zeros((1, NUM_TOPICS))
            for i in range(2, len(val)):
                s = val[i].split(":")
                pr[0, i-2] =float(s[1])
            dict[(userID,queryID)] = pr
    return dict
def readIndri():
    dict_indri = {}
    for uqPair in INDRI_LIST:
        user = uqPair[0]
        query = uqPair[1]
        path = "../data/indri-lists/"+ str(user)+"-"+ str(query)+".results.txt"
        indri_score = {}
        with open(path, "r") as myfile:
            for line in myfile:
                val = line.strip().split(" ")
                indri_score[int(val[2])-1] = float(val[4])
        dict_indri[uqPair] = indri_score
    return dict_indri


def writeFile(uqPair, score_list):
    rank = 1
    usr = uqPair[0]
    query = uqPair[1]
    sorted_scList = sorted(score_list, key=score_list.get, reverse=True)
    with open("../data/result.txt", "a") as myfile:
        for docID in sorted_scList:
            myfile.write(str(usr)+"-"+str(query) + " Q0 " + str(docID+1) + " " + str(rank) + " " + str(score_list.get(docID)) + " run-1\n")
            rank += 1
    myfile.close()


def TSPRHelper(trans, p_t, parameter):
    [r, p_0] = initHelper()
    its = 0
    diff = 1
    while diff > float(parameter.get("eps")) and its < float(parameter.get("iter")):
        [next_r, diff] = iterHelperTS(r, trans, p_0, p_t, float(parameter.get("alpha")), float(parameter.get("beta")))
        r = np.matrix(next_r)
        its += 1
    return r
def initHelper():
    r = np.random.np.random.dirichlet(np.ones(NUM_DOCS))
    r = r.reshape(NUM_DOCS,1)
    p_0 = np.ones((NUM_DOCS, 1))
    p_0 = p_0 / p_0.size
    return r,p_0
def iterHelper(r, trans, p_0, alpha):
    next_r = alpha*trans.transpose().dot(r) + (1-alpha)*p_0
    diff = np.sum(np.abs(r-next_r))
    return next_r, diff
def iterHelperTS(r, trans, p_0, p_t, alpha, beta):
    next_r = alpha*trans.transpose().dot(r) + beta*p_t + (1-alpha-beta)*p_0
    diff = np.sum(np.abs(r-next_r))
    return next_r, diff

# GPR calculation
def PR_r(trans, parameter):
    [r, p_0] = initHelper()
    its = 0
    diff = 1
    while diff > float(parameter.get("eps")) and its < float(parameter.get("iter")):
        [next_r, diff] = iterHelper(r, trans, p_0, float(parameter.get("alpha")))
        r = np.matrix(next_r)
        its += 1
    return r
# QTSPR and PTSPR calculation
def TSPR_r(trans, dict_pt, dict_prior, parameter):
    for topic in sorted(dict_pt):
        r_t = TSPRHelper(trans, dict_pt.get(topic), parameter)
        if topic == 1:
            R = np.matrix(r_t.transpose())
        else:
            R = np.concatenate((R, r_t.transpose()))
    dict_tspr = {}
    for uqPair in dict_prior:
        r_tspr = dict_prior.get(uqPair).dot(R)
        r_tspr = r_tspr.transpose()
        dict_tspr[uqPair] = r_tspr
    return dict_tspr

# Combine Indri and PageRank score
def getScList(parameter, indri_score, r):
    score_list = {}
    type = parameter.get("weighting scheme")
    # maxval = r.max()
    # scale = 1000
    # while maxval*scale < 1:
    #     scale *= 10
    if type == "CM":
        # r = (r - r.mean()) / r.var()
        # scaledr = r
        scaledr = (r - r.min()) / (r.max()-r.min())
    else:
        scaledr = r * float(parameter.get("scale"))
    # print scaledr.max(), scaledr.min()
    for docID in indri_score:
        indri = indri_score.get(docID)
        pagerank = scaledr[docID, 0]
        if type == "NS":
            newScore = pagerank
        elif type == "WS":
            newScore = weightSum(indri, pagerank, float(parameter.get("weight")))
        elif type == "CM":
            newScore = cmSum(indri, pagerank,float(parameter.get("weight")))
        else:
            newScore = 0
            print "Wrong weighting scheme"
        score_list[docID] = newScore
    return score_list
def weightSum(indri, pagerank, weight):
    return weight*indri + (1-weight)*pagerank
def cmSum (indri, pagerank, weight):
    # a = 1/(1+weight*weight)
    # a = a/pow(2, indri)
    # b = (weight*weight)/(1+weight*weight)
    # b = b/pagerank
    return weight*indri - (1-weight)*pagerank*indri
    # return weight*indri + (1-weight)*math.log(pagerank)
    # return 1/(a+b)


def GPR(parameter):
    global start_time
    trans = readTrans()
    print "Transition Matrix Read.."
    start_time = time.time()
    r = PR_r(trans, parameter)
    print "r calculated.."
    dict_indri = readIndri()
    for uqPair in dict_indri:
        score_list = getScList(parameter, dict_indri.get(uqPair), r)
        writeFile(uqPair, score_list)
    print "Finished"
def TSPR(isUserPrior, parameter):
    global start_time
    trans = readTrans()
    print "Transition Matrix Read.."
    dict_pt = readTopic()
    dict_prior = readPrior(isUserPrior)
    start_time = time.time()
    dict_tspr = TSPR_r(trans, dict_pt, dict_prior, parameter)
    print "r calculated.."
    dict_indri = readIndri()
    for uqPair in dict_indri:
        score_list = getScList(parameter, dict_indri.get(uqPair), dict_tspr.get(uqPair))
        writeFile(uqPair, score_list)
    print "Finished"

def run():
    parameter = readParam()
    print "Parameter: ", parameter
    if os.path.isfile("../data/result.txt"):
        os.remove("../data/result.txt")
    alg = parameter.get("algorithm")
    if alg == "GPR":
        GPR(parameter)
    elif alg == "QTSPR":
        TSPR(False, parameter)
    elif alg == "PTSPR":
        TSPR(True, parameter)
    else:
        print "wrong algorithm"
    duration = time.time()-start_time
    print "Running Time:", duration

def sampleFile():
    parameter = readParam()
    trans = readTrans()
    gpr = PR_r(trans, parameter)
    print len(gpr)
    with open("../data/GPR-10.txt", "a") as myfile:
        for i in range(len(gpr)):
            myfile.write(str(i+1)+" "+ str(gpr[i,0]) + "\n")
    dict_pt = readTopic()
    dict_prior = readPrior(False)
    dict_tspr = TSPR_r(trans, dict_pt, dict_prior, parameter)
    qtspr = dict_tspr.get((2,1))
    with open("../data/QTSPR-U2Q1-10.txt", "a") as myfile:
        for i in range(len(qtspr)):
            myfile.write(str(i+1)+" "+ str(qtspr[i,0]) + "\n")
    dict_prior = readPrior(True)
    dict_tspr = TSPR_r(trans, dict_pt, dict_prior, parameter)
    ptspr = dict_tspr.get((2,1))
    with open("../data/PTSPR-U2Q1-10.txt", "a") as myfile:
        for i in range(len(ptspr)):
            myfile.write(str(i+1)+" "+ str(ptspr[i,0]) + "\n")

if __name__ == "__main__":
    run()
    # sampleFile()