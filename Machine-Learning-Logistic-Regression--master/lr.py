'''Created on Mar20, 2014
Logistic  regression classify with (Random) Gradient Ascent method
@author: Aidan
'''

from numpy import *
from object_json import *
from copy import *
import pdb


class logisticRegres(object):
    def __init__(self, classifierArray=None, **args):
        '''classifierArray is (1,m+1)numpy array, m is the
        feture number of sample'''
        obj_list = inspect.stack()[1][-2]
        self.__name__ = obj_list[0].split('=')[0].strip()

        self.classifierArray = classifierArray

    def jsonDumpsTransfer(self):
        '''essential transformation to Python basic type in order to
        store as json. dumps as objectname.json if filename missed '''
        # pdb.set_trace()
        self.classifierArray = self.classifierArray.tolist()

    def jsonDumps(self, filename=None):
        '''dumps to json file'''
        self.jsonDumpsTransfer()
        if not filename:
            jsonfile = self.__name__ + '.json'
        else:
            jsonfile = filename
        objectDumps2File(self, jsonfile)

    def jsonLoadTransfer(self):
        '''essential transformation to object required type, such as numpy matrix
        call this function after newobject = objectLoadFromFile(jsonfile)'''
        # pdb.set_trace()
        self.classifierArray = array(self.classifierArray)

    def getClassifierArray(self):
        return self.classifierArray

    def setClassifierArray(self, classifierArray):
        self.classifierArray = deepcopy(classifierArray)

    def __sigmoid(self, inX):
        return 1.0 / (1 + exp(-inX))

    def classifyArray(self, dataToClassList):
        '''dataToClassListis (1,n)numpy array, n indicates the sample number, each sampe is (1,m)list
        for example [[1,2],],[[1,2],[3,4]]'''
        dataToClassMat = mat(dataToClassList)
        # pdb.set_trace()
        n, m = dataToClassMat.shape
        estClassMat = self.__sigmoid(self.classifierArray * dataToClassMat.T)
        # pdb.set_trace()
        estClassMat[estClassMat > 0.5] = 1.0
        estClassMat[estClassMat <= 0.5] = 0.0
        return estClassMat

    def classifySample(self, dataToClass):
        '''dataToClass is a sample. for example [1,2]'''
        prob = self.__sigmoid(sum(self.classifierArray * dataToClass))

        if prob > 0.5:
            return 1
        else:
            return 0

    def __gradAscent(self, dataMatIn, classLabels, numIter=500):
        '''the return weights is (1,m+1)numpy array,
        m indicates the sample feturen number'''
        dataMatrix = mat(dataMatIn)  # convert to NumPy matrix
        labelMat = mat(classLabels).transpose()  # convert to NumPy matrix
        m, n = shape(dataMatrix)
        alpha = 0.001  # step
        weights = mat(ones((n, 1)))  # the default value is 1.0
        for k in range(numIter):  # heavy on matrix operations
            h = self.__sigmoid(dataMatrix * weights)  # matrix mult
            error = (labelMat - h)  # vector subtraction
            weights = weights + alpha * dataMatrix.transpose() * error  # matrix mult
        return array((weights.T))[0]  # return weights array

    def __randomGradAscent(self, dataMatIn, classLabels, numIter=150):
        '''the return weights is (1,m+1)numpy array,
        m indicates the sample feturen number'''
        dataMatrix = array(dataMatIn)
        m, n = shape(dataMatrix)
        weights = ones(n)  # initialize to all ones
        for j in range(numIter):
            dataIndex = range(m)
            for i in range(m):
                alpha = 4 / (1.0 + j + i) + 0.0001  # apha decreases with iteration, does not
                randIndex = int(random.uniform(0, len(dataIndex)))  # go to 0 because of the constant
                h = self.__sigmoid(sum(dataMatrix[randIndex] * weights))
                error = classLabels[randIndex] - h
                # pdb.set_trace()
                weights = weights + alpha * error * dataMatrix[randIndex]
                del (dataIndex[randIndex])  # in case samples may be choosed repeatly
        return weights

    def train(self, dataMatrix, classLabels, solve='randomGrad', numIter=150):
        '''LR classify train mthod, the mtrain surpport two solve method,
        gradAscent('Grad') and randomGradAscent('randomGrad'), the default
        solver is randomGradAscent('randomGrad')'''

        # pdb.set_trace()
        if solve == 'randomGrad':
            self.classifierArray = self.__randomGradAscent(dataMatrix, classLabels, numIter)
            return self.classifierArray
        elif solve == 'Grad':
            self.classifierArray = self.__gradAscent(dataMatrix, classLabels, numIter)
            return self.classifierArray
        else:
            print "current LR surpport gradAscent('Grad')and randomGradAscent('randomGrad')"
            return None


def plotBestFit(weights, dataMat, labelMat):
    import matplotlib.pyplot as plt
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = [];
    ycord1 = []
    xcord2 = [];
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]);
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]);
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1');
    plt.ylabel('X2');
    plt.show()


if __name__ == '__main__':

    def loadDataSet():
        dataMat = [];
        labelMat = []
        fr = open('testSet.txt')
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
        return dataMat, labelMat


    dataMat, labelMat = loadDataSet()
    testLR_random = logisticRegres()
    r_weights = testLR_random.train(dataMat, labelMat, numIter=400)
    est05 = testLR_random.classifyArray(dataMat[0:5])
    print 'est05: ', est05
    # pdb.set_trace()
    print 'random weights: ', r_weights
    plotBestFit(r_weights, dataMat, labelMat)

    weights = testLR_random.train(dataMat, labelMat, 'Grad', numIter=500)
    print 'weights: ', weights
    plotBestFit(weights, dataMat, labelMat)
