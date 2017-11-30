# library inclusion
import numpy as np
import csv as csv

#### from blog, need to be removed
def classify(inX, dataSet, labels, k):
    inX=np.mat(inX)
    dataSet=np.mat(dataSet)
    labels=np.mat(labels)
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = np.array(diffMat)**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[0,sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=np.operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#normalize the original data to grey-scale
def toOne(data):
    data = np.mat(data)
    row,col = np.shape(data)
    for i in xrange(row):
        for j in xrange(col):
            if data[i,j] != 0:
                data[i, j] = 1
    return data


# convert string data to int data
def toInt(data):
    data = np.mat(data)
    row, col = np.shape(data)
    res = np.zeros((row, col))

    for i in xrange(row):
        for j in xrange(col):
            res[i, j] = int(data[i, j])
    return res


# Preprocessing
def loadData(fname):
    '''
    Params: file name
    Return: 1. image pixel data in greyscale(1,0 only) 2.corresponding labels
    '''
    raw = []
    with open(fname) as fil:
        for row in csv.reader(fil):
            raw.append(row)
        raw = np.array(raw)  # remove first row (descriptor)
        raw = raw[1:]
        label = raw[:, 0]
        data = raw[:, 1:]
        fil.close()
        return toOne(toInt(data)), toInt(label)
# load test data
def loadTestData(fname):
    raw = []
    with open(fname) as fil:
        for row in csv.reader(fil):
            raw.append(row)
        raw = np.array(raw)
        raw = raw[1:]
        fil.close()
        return toOne(toInt(raw))

# load test labels
def loadTestLabels(fname):
    raw = []
    with open(fname) as fil:
        for row in csv.reader(fil):
            raw.append(row)
        raw = np.array(raw)
        raw = raw[1:]
        fil.close()
        return toInt(raw)
#save the result in a csv file
def saveResult(result):
    with open('result.csv','wb') as myFile:
        myWriter=csv.writer(myFile)
        for i in result:
            tmp=[]
            tmp.append(i)
            myWriter.writerow(tmp)



# main function:
if __name__ == '__main__':
    print 'loading training data and label'
    train_data, train_label = loadData('train.csv')
    print 'complete'
    print 'loading test data and label'
    testData = loadTestData('test.csv')
    testLabel = loadTestLabels('knn_benchmark.csv')
    print 'complete'
    row,col = np.shape(testData)
    print 'dimension of test data: col:' + str(col) + ' row: ' + str(row)
    error = 0
    res = list()
    k = 5
    print 'start prediction'
    for test in xrange(row):
        prediction = classify(testData[test], train_data, train_label, k)
        res.append(prediction)
        if (prediction != testLabel[0, test]): error += 1

    print "\nthe total number of errors is: %d",error
    print "\nthe total error rate is: %f",(error / float(row))

    with open('result.csv', 'wb') as myFile:
        writer = csv.writer(myFile)
        for i in res:
            tmp = []
            tmp.append(i)
            writer.writerow(tmp)
