import math

def accuracy(predict, label):
    count = 0
    for i in xrange(len(predict)):
        if predict[i] == label[i]:
            count += 1
    return float(count)/len(predict)

def rmse (predict, label):
    count = 0
    for i in xrange(len(predict)):
        count += (predict[i]-label[i]) ** 2
    count = float(count)/len(predict)
    return math.sqrt(count)

if __name__ == "__main__":
    lst_pred = [1,1,1,1]
    lst_label = [2,1,1,1]
    print rmse(lst_pred, lst_label)