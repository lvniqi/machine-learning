# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:54:53 2016

@author: lvniqi
"""
from math import log
from copy import copy, deepcopy

import operator


class DecisionTree:
    def __init__(self, data=None, label=None):
        pass

    @staticmethod
    def calcShannonEnt(data):
        '''
        用于计算香农熵
        :param data: 数据集
        :return: 香农熵
        '''
        data_len = len(data)
        lables = {}
        for currentV in data:
            # 最后一个为符号
            currentLabel = currentV[-1]
            # 如果在label中符号计数
            if currentLabel not in lables.keys():
                lables[currentLabel] = 0
            lables[currentLabel] += 1
        shannonEnt = 0.0
        for key in lables:
            prob = 1.0 * lables[key] / data_len
            shannonEnt -= prob * log(prob, 2)
        return shannonEnt

    @staticmethod
    def splitData(data, pos, value):
        '''
        用于分割数据
        :param data: 原始数据集
        :param pos: 检测位置
        :param value: 检测值
        :return: 删除pos属性后的检测成功的数据
        '''
        # 如果满足值相等，则将数据保存
        retData = deepcopy([x for x in data if x[pos] == value])
        # 弹出这一列
        for x in retData:
            x.pop(pos)
        return retData

    @staticmethod
    def chooseBestFeatureToSplit(data):
        '''
        选择最好的一个属性老分割数据集
        :param data: 原始数据集
        :return: 最好的属性
        '''
        # 参数数量
        feature_len = len(data[0]) - 1
        # 基准信息量
        base_entropy = DecisionTree.calcShannonEnt(data)
        # 信息增益
        best_info_gain = 0.0
        # 最佳列
        best_feature = -1
        for pos in range(feature_len):
            current_feature_list = [currentV[pos] for currentV in data]
            unique_value = set(current_feature_list)
            new_entropy = 0
            # 计算各个数值的信息熵
            for value in unique_value:
                # 分割数据集
                sub_data = DecisionTree.splitData(data, pos, value)
                prob = 1.0 * len(sub_data) / len(data)
                new_entropy += prob * DecisionTree.calcShannonEnt(sub_data)
            info_gain = base_entropy - new_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = pos
        return best_feature

    @staticmethod
    def majorityCnt(result_list):
        '''
        投票表决 在无法划分时使用
        :param result_list: 结果集
        :return: 结果中数量较多的判断为正确的
        '''
        class_count = {}
        for vote in result_list:
            if vote not in class_count.keys():
                class_count[vote] = 0
            class_count[vote] += 1
        sorted_class_count = sorted(class_count.items(), key=lambda x: x[1], reverse=True)
        return sorted_class_count[0][0]

    @staticmethod
    def createTree(data, labels):
        '''
        构建决策树
        :param data:输入数据集
        :param labels: 属性集
        :return:
        '''
        result_list = [item[-1] for item in data]
        # 如果为同一类，则直接返回这一类
        if result_list.count(result_list[0]) == len(result_list):
            return result_list[0]
        # 如果没有属性，则使用投票表决的方法
        if len(data[0]) == 1:
            return DecisionTree.majorityCnt(result_list)
        best_feat = DecisionTree.chooseBestFeatureToSplit(data)
        best_feat_label = labels[best_feat]
        tree = {best_feat_label: {}}
        labels.pop(best_feat)
        # 得到这个属性的所有值
        feats = [item[best_feat] for item in data]
        unique_val = set(feats)
        for value in unique_val:
            sub_labels = copy(labels)
            sub_data = DecisionTree.splitData(data, best_feat, value)
            tree[best_feat_label][value] = DecisionTree.createTree(sub_data, sub_labels)
        return tree


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
        return dataSet, labels


    (data, labels) = createDataSet()
    print DecisionTree.splitData(data, 0, 1)
    print DecisionTree.calcShannonEnt(data)
    print DecisionTree.chooseBestFeatureToSplit(data)

    print DecisionTree.createTree(data, labels)

    # test_vote_list = ['no','no','no','yes','yes']
    # print  DecisionTree.majorityCnt(test_vote_list)
