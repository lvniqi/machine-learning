# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:54:53 2016

@author: lvniqi
"""
from math import log
from copy import deepcopy

class DecisionTree:
    def __init__(self,data = None,label = None):
        pass

    @staticmethod
    def calcShannonEnt(data):
        data_len = len(data)
        lables = {}
        for currentV in data:
            #最后一个为符号
            currentLabel = currentV[-1]
            #如果在label中符号计数
            if currentLabel not in lables.keys():
                lables[currentLabel] = 0
            lables[currentLabel] += 1
        shannonEnt = 0.0
        for key in lables:
            prob = 1.0*lables[key]/data_len
            shannonEnt -= prob*log(prob,2)
        return  shannonEnt

    @staticmethod
    def splitData(data,pos,value):
        #如果满足值相等，则将数据保存
        retData = deepcopy([x for x in data if x[pos] == value])
        #弹出这一列
        for x in retData:
            x.pop(pos)
        return retData

    @staticmethod
    def chooseBestFeatureToSplit(data):
        #参数数量
        feature_len = len(data[0])-1
        #基准信息量
        base_entropy = DecisionTree.calcShannonEnt(data)
        #信息增益
        best_info_gain = 0.0
        #最佳列
        best_feature = -1
        for pos in range(feature_len):
            current_feature_list = [currentV[pos] for currentV in data]
            unique_value = set(current_feature_list)
            new_entropy = 0
            #计算各个数值的信息熵
            for value in unique_value:
                #分割数据集
                sub_data = DecisionTree.splitData(data,pos,value)
                prob = 1.0 * len(sub_data) / len(data)
                new_entropy += prob*DecisionTree.calcShannonEnt(sub_data)
            info_gain = base_entropy - new_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = pos
        return best_feature
if __name__ == "__main__":
    def createDataSet():
        dataSet = [
            [1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no'],
        ]
        labels = [
            'no surfacing',
            'flippers',
        ]
        return dataSet,labels
    (data,labels) = createDataSet()
    print DecisionTree.splitData(data,0,1)
    print DecisionTree.calcShannonEnt(data)
    print DecisionTree.chooseBestFeatureToSplit(data)