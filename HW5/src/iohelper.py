import json
import re

def readTrain():
    # read json file
    lst_text = []
    lst_stars = []
    with open('../data/yelp_reviews_train.json', 'r') as f:
        count = 0
        for line in f:
            review = json.loads(line)
            lst_text.append(review['text'].encode('utf-8'))
            # lst_text.append(review['text'])
            lst_stars.append(review['stars'])
            # count += 1
            # if count > 5000:
            #     break
    print "TRAIN REVIEWS READ!"
    return lst_text, lst_stars

def readDev():
    # read json file
    lst_text = []
    with open('../data/yelp_reviews_dev.json', 'r') as f:
        count = 0
        for line in f:
            review = json.loads(line)
            lst_text.append(review['text'].encode('utf-8'))
            # count += 1
            # if count > 1000:
            #     break
    print "DEV REVIEWS READ!"
    return lst_text

def readTest():
    # read json file
    lst_text = []
    with open('../data/yelp_reviews_test.json', 'r') as f:
        count = 0
        for line in f:
            review = json.loads(line)
            lst_text.append(review['text'].encode('utf-8'))
            # count += 1
            # if count > 1000:
            #     break
    print "TEST REVIEWS READ!"
    return lst_text

def writeDevPred(lst_stars_hard, lst_stars_soft):
    with open("../data/dev.txt", "w") as myfile:
        for i in xrange(len(lst_stars_hard)):
            myfile.write(str(lst_stars_hard[i]) + " " + str(lst_stars_soft[i]) + '\n')

def writeTestPred(lst_stars_hard, lst_stars_soft):
    with open("../data/test.txt", "w") as myfile:
        for i in xrange(len(lst_stars_hard)):
            myfile.write(str(lst_stars_hard[i]) + " " + str(lst_stars_soft[i]) + '\n')


def readStopWords():
    stop_words = [k.rstrip() for k in open("../data/stopword.list","r").readlines()]
    stop_words = set(stop_words)
    return stop_words


# helper function to convert any decoded JSON object from using unicode strings to UTF-8-encoded byte strings
def byteify(input):
    if isinstance(input, dict):
        return {byteify(key): byteify(value)
                for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input

# helper func not in use
def write_review (datatype, lst_text):
    with open("../data/review_" + datatype, "a") as myfile:
        for text in lst_text:
            myfile.write(text + '\n')

# helper func not in use
def read_review (datatype):
    lst_text = [k.rstrip() for k in open("../data/review_" + datatype,"r").readlines()]
    return lst_text

# if __name__ == "__main__":
    # text = "I'm the adf , cd2 ? acds # agiwe."
    # stop_words = readStopWords()
    # print preprocess(text, stop_words)