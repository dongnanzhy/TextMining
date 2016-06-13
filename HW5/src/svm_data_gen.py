import preprocess
import iohelper

def writeTrain(X, lst_label, file):
    with open(file, 'w') as f:
        for x in xrange(X.shape[0]):
            s = ""
            s += str(lst_label[x])
            s += " "
            tmp = X[x].toarray().flatten()
            col = tmp.nonzero()[0]
            data = tmp[col]
            for y, z in zip(col, data):
                s += str(y+1)
                s += ":"
                s += str(z)
                s += " "
            s += "\n"
            f.write(s)
    return

def writeTest(X, file):
    with open(file, 'w') as f:
        for x in xrange(X.shape[0]):
            s = "0 "
            tmp = X[x].toarray().flatten()
            col = tmp.nonzero()[0]
            data = tmp[col]
            for y, z in zip(col, data):
                s += str(y+1)
                s += ":"
                s += str(z)
                s += " "
            s += "\n"
            f.write(s)
    return

def genData():
    lst_train_text, lst_train_stars = iohelper.readTrain()
    lst_dev_text = iohelper.readDev()

    stop_words = iohelper.readStopWords()
    lst_train_text = [preprocess.preprocess(text, stop_words) for text in lst_train_text]
    lst_train_BOW = [preprocess.toBOW(text) for text in lst_train_text]
    lst_dev_text = [preprocess.preprocess(text, stop_words) for text in lst_dev_text]
    lst_dev_BOW = [preprocess.toBOW(text) for text in lst_dev_text]
    print "PREPROCESS FINISHED!"

    train_data, dev_data, _ = preprocess.getData(lst_train_BOW, lst_dev_BOW, [], 2000, isCF=False, isDF=True)
    print "DATA MATRIX GENERATED!"

    writeTrain(train_data, lst_train_stars, "../data/svm_train")
    writeTest(dev_data, "../data/svm_dev")
    # writeTest(train_data, "../data/svm_train_test")

# for online test
def modifyFormat():
    lst_stars = read_pred("../data/svm_dev_result")
    write_pred("../data/dev_svm_pred", lst_stars)
    # lst_stars = read_pred("../data/train_svm_result")
    # write_pred("../data/train_svm_pred", lst_stars)

# helper func for online test
def write_pred(file, lst_star):
    with open(file, "a") as myfile:
        for star in lst_star:
            myfile.write(str(star) + " " + str(0.0) + '\n')

# helper func for online test
def read_pred(file):
    lst_star = [k.rstrip() for k in open(file,"r").readlines()]
    return lst_star


if __name__ == "__main__":
    # genData()
    modifyFormat()

