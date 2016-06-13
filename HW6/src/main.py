import numpy as np
import pmf
import multiLR
import util
import argparse

def run():
    parser = argparse.ArgumentParser(description = "collaborative ranking")
    parser.add_argument("-f", help = "number of features", type = int, choices = [10,20,50,100])
    parser.add_argument("-m", help = "predicting method", type = int, choices = [1,2])
    args = parser.parse_args()
    print args

    '''
    # Get Feature Vector U and V
    '''
    [U, V] = pmf.pmf_GD(args.f)
    pmf.saveLatentMat(U, "U")
    pmf.saveLatentMat(V, "V")

    '''
    # Get training and test data
    '''
    U = pmf.loadLatentMat("U")
    V = pmf.loadLatentMat("V")
    util.genTrainData(U, V)
    util.genTestData(U, V, isDev=True)

    '''
    # Perform LR
    '''
    W = multiLR.train()
    # multiLR.pred(W, isDev=True)
    # multiLR.pred_direct(W, isDev=True)

    if args.m == 2:
        multiLR.pred_direct(W, isDev=True)
    else:
        multiLR.pred(W, isDev=True)
        util.genResult(isDev=True)

    '''
    # Get final prediction based on binary classifications
    '''
    # util.genResult(isDev=True)

    '''
    # Get PMF result
    '''
    U = pmf.loadLatentMat("U")
    V = pmf.loadLatentMat("V")
    pmf.predict(U, V, isDev=False)

def corpus_explore():
    total_instance = 0
    num_positive_label = 0
    num_negative_label = 0
    with open("../data/rank_train.data", 'r') as f:
        for line in f:
            val = line.strip().split(" ")
            if int(val[0]) == 1:
                num_positive_label += 1
            else:
                num_negative_label += 1
            total_instance += 1
    print "number of instance: ", total_instance
    print "number of positive label: ", num_positive_label
    print "number of negative label: ", num_negative_label

    user_dict = util.readTrainList()
    lst_movie = user_dict[1234]
    xx = 0
    yy = 0
    for item in lst_movie:
        rate = item[1]
        if rate == 1 or rate == 5:
            if rate == 1:
                xx += 1
            else:
                yy += 1
    print "number of examples for user 1234 is: ", 2*xx*yy

    lst_movie = user_dict[4321]
    xx = 0
    yy = 0
    for item in lst_movie:
        rate = item[1]
        if rate == 1 or rate == 5:
            if rate == 1:
                xx += 1
            else:
                yy += 1
    print "number of examples for user 4321 is: ", 2*xx*yy

    return

if __name__ == "__main__":
    run()
    # corpus_explore()