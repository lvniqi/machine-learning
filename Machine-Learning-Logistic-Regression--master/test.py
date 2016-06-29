'''Created on Mar20, 2014
Logistic  regression classify with (Random) Gradient Ascent method test
@author: Aidan
'''
from numpy import *
from object_json import *

import pdb


def loadDataSet(fileName):  # general function to parse tab -delimited floats
    dataSet = [];
    Labels = []
    with open(fileName) as fr:
        for line in fr.readlines():
            currLine = line.strip().split('\t')
            lineArr = [1.0, ]
            print lineArr
            for i in range(21):
                lineArr.append(float(currLine[i]))
            dataSet.append(lineArr)
            Labels.append(float(currLine[21]))
    return dataSet, Labels


def LRTest():
    dataArr, labelArr = loadDataSet('horseColicTraining.txt')
    # datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    numIt = 150
    filename = 'LRClassifier' + repr(numIt) + '.json'
    try:
        LRClassifier = objectLoadFromFile(filename)
        LRClassifier.jsonLoadTransfer()
        print 'load LRClassifier successfully'
    except IOError, ValueError:
        print 'LRClassifier doesnt exist. train first'
        from lr import *
        LRClassifier = logisticRegres()
        classifierArray = LRClassifier.train(dataArr, labelArr, numIter=numIt)
        LRClassifier.jsonDumps(filename)

        LRClassifier.jsonLoadTransfer()

    # train error
    classest = LRClassifier.classifyArray(dataArr)
    labelMat = mat(labelArr)
    errArr = mat(ones(labelMat.shape))
    # pdb.set_trace()
    errSum = errArr[classest != labelMat].sum()
    print 'the count of train pridict error is', errSum, ',total train samples ', len(dataArr)

    # TEST
    testArr, testlabelArr = loadDataSet('horseColicTest.txt')
    classestMat = LRClassifier.classifyArray(testArr)
    testlabelMat = mat(testlabelArr)
    errArr = mat(ones(testlabelMat.shape))
    # pdb.set_trace()
    errSum = errArr[classestMat != testlabelMat].sum()
    print 'the count of test pridict error is', errSum, errSum / 67.0


    # print "the test error rate is: %2.2f%%" % ((float(errorCount)/m)*100 )


if __name__ == '__main__':
    LRTest()
