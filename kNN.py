# -*- coding: utf-8 -*-
"""
Created on Sat May 28 17:53:54 2016

@author: lvniqi
"""

import numpy as np
import operator
import os


#  knn分类器
def kNNClassify(new_input, data_set, labels, k):
    num_samples = data_set.shape[0]  # shape[0] stands for the num of row

    # 第一步 计算欧氏距离
    # tile(A, reps): Construct an array by repeating A reps times  
    # the following copy num_samples rows for dataSet
    diff = np.tile(new_input, (num_samples, 1)) - data_set  # Subtract element-wise
    squaredDiff = diff ** 2  # squared for the subtract
    squaredDist = np.sum(squaredDiff, axis=1)  # sum is performed by row
    distance = squaredDist ** 0.5

    #  step 2: sort the distance
    #  argsort() returns the indices that would sort an array in a ascending order
    sortedDistIndices = np.argsort(distance)

    classCount = {}  # define a dictionary (can be append element)
    for i in xrange(k):
        #  step 3: choose the min k distance
        voteLabel = labels[sortedDistIndices[i]]

        #  step 4: count the times labels occur
        # when the key voteLabel is not in dictionary classCount, get()  
        # will return 0  
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

        #  step 5: the max voted class will return
    max_count = 0
    for key, value in classCount.items():
        if value > max_count:
            max_count = value
            max_index = key

    return max_index


# convert image to vector  
def img2vector(filename):
    rows = 32
    cols = 32
    img_vector = np.zeros((1, rows * cols))
    file_in = open(filename)
    for row in xrange(rows):
        line_str = file_in.readline()
        for col in xrange(cols):
            img_vector[0, row * 32 + col] = int(line_str[col])

    return img_vector


# load dataSet  
def loadDataSet():
    ## step 1: Getting training set  
    print "---Getting training set..."
    dataSetDir = 'E:/WORKSPACE/machine-learning/data/'
    trainingFileList = os.listdir(dataSetDir + 'trainingDigits')  # load the training set
    numSamples = len(trainingFileList)

    train_x = np.zeros((numSamples, 1024))
    train_y = []
    for i in xrange(numSamples):
        filename = trainingFileList[i]

        # get train_x  
        train_x[i, :] = img2vector(dataSetDir + 'trainingDigits/%s' % filename)

        # get label from file name such as "1_18.txt"  
        label = int(filename.split('_')[0])  # return 1
        train_y.append(label)

        ## step 2: Getting testing set
    print "---Getting testing set..."
    testingFileList = os.listdir(dataSetDir + 'testDigits')  # load the testing set
    numSamples = len(testingFileList)
    test_x = np.zeros((numSamples, 1024))
    test_y = []
    for i in xrange(numSamples):
        filename = testingFileList[i]

        # get train_x  
        test_x[i, :] = img2vector(dataSetDir + 'testDigits/%s' % filename)

        # get label from file name such as "1_18.txt"  
        label = int(filename.split('_')[0])  # return 1
        test_y.append(label)

    return train_x, train_y, test_x, test_y


# test hand writing class  
def testHandWritingClass():
    #  第一步 加载数据
    print "step 1: load data..."
    (train_x, train_y, test_x, test_y) = loadDataSet()

    #  step 2: training...
    print "step 2: training..."
    pass

    #  step 3: testing
    print "step 3: testing..."
    numTestSamples = test_x.shape[0]
    matchCount = 0
    for i in xrange(numTestSamples):
        predict = kNNClassify(test_x[i], train_x, train_y, 3)
        if predict == test_y[i]:
            matchCount += 1
    accuracy = float(matchCount) / numTestSamples

    ## step 4: show the result  
    print "step 4: show the result..."
    print 'The classify accuracy is: %.2f%%' % (accuracy * 100)


if __name__ == '__main__':
    testHandWritingClass()