import iohelper
import multiLR
import preprocess

def run(num_feature, isCF, isDF, isCustom):
    lst_train_text, lst_train_stars = iohelper.readTrain()
    lst_dev_text = iohelper.readDev()
    lst_test_text = iohelper.readTest()

    stop_words = iohelper.readStopWords()
    lst_train_text = [preprocess.preprocess(text, stop_words) for text in lst_train_text]
    lst_train_BOW = [preprocess.toBOW(text) for text in lst_train_text]
    lst_dev_text = [preprocess.preprocess(text, stop_words) for text in lst_dev_text]
    lst_dev_BOW = [preprocess.toBOW(text) for text in lst_dev_text]
    lst_test_text = [preprocess.preprocess(text, stop_words) for text in lst_test_text]
    lst_test_BOW = [preprocess.toBOW(text) for text in lst_test_text]
    print "PREPROCESS FINISHED!"

    if isCustom:
        train_data, dev_data, test_data = preprocess.getData_custom(lst_train_BOW, lst_dev_BOW, lst_test_BOW, num_feature, isCF, isDF)
    else:
        train_data, dev_data, test_data = preprocess.getData(lst_train_BOW, lst_dev_BOW, lst_test_BOW, num_feature, isCF, isDF)
    print "DATA MATRIX GENERATED!"

    W = multiLR.train(train_data, lst_train_stars)

    print "START PREDICT ON DEVELOPMENT DATA!"
    lst_dev_hard, lst_dev_soft = multiLR.pred(dev_data, W)
    iohelper.writeDevPred(lst_dev_hard, lst_dev_soft)

    print "START PREDICT ON TEST DATA!"
    lst_test_hard, lst_test_soft = multiLR.pred(test_data, W)
    iohelper.writeTestPred(lst_test_hard, lst_test_soft)

def corpus_explore():
    lst_train_text, lst_train_stars = iohelper.readTrain()
    stop_words = iohelper.readStopWords()
    lst_train_text = [preprocess.preprocess(text, stop_words) for text in lst_train_text]
    lst_train_BOW = [preprocess.toBOW(text) for text in lst_train_text]
    lst_top, dict_CF =  preprocess.getCF_Dict(lst_train_BOW, 9)
    print "total number of words: ", len(dict_CF)
    for word in lst_top:
        print word, ": ", dict_CF[word]
    stars_count = [0,0,0,0,0]
    for star in lst_train_stars:
        stars_count[star-1] += 1
    for i in xrange(len(stars_count)):
        print "number of examples: ", stars_count[i], "percentage: ", stars_count[i]/float(1255353)


if __name__ == "__main__":
    run(6000, isCF=True, isDF=False, isCustom=True)
    # corpus_explore()