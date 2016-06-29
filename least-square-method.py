# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:54:53 2016

@author: lvniqi
"""
import numpy as np
import random
import matplotlib.pyplot as plt


def randData():
    # 生成曲线上各个点
    x = np.arange(-1, 1, 0.02)
    y = [2 * a + 3 for a in x]  # 直线
    #     y = [((a*a-1)*(a*a-1)*(a*a-1)+0.5)*np.sin(a*2) for a in x]  # 曲线
    xa = [];
    ya = []
    # 对曲线上每个点进行随机偏移
    for i in range(len(x)):
        d = np.float(random.randint(90, 120)) / 100
        ya.append(y[i] * d)
        xa.append(x[i] * d)
    return xa, ya


def hypfunc(x, A):
    # 输入：x 横坐标数值， A 多项式系数 [a0,a1,...,an-1]
    # 返回 y = hypfunc(x)
    return np.sum(A[i] * (x ** i) for i in range(len(A)))


# 使用 θ = (X.T*X + λI)^-1 * X.T * y求解直线参数
# 该函数会在X的前面添加偏移位X0 = 1
def LS_line(X, Y, lam=0.01):
    x = np.array(X)
    x = np.vstack((np.ones((len(x),)), x))  # 往上面添加X0
    x = np.mat(x).T  # (m,n)
    y = np.mat(Y).T  # (m,1)
    M, N = x.shape
    I = np.eye(N, N)  # 单位矩阵

    theta = ((x.T * x + lam * I) ** -1) * x.T * y  # 核心公式
    theta = np.array(np.reshape(theta, len(theta)))[0]
    return theta  # 返回一个一维数组


# 使用随机梯度下降法求解最小二参数:
# alpha 迭代步长（固定步长），epslion 收敛标准
def LS_sgd(X, Y, alpha=0.1, epslion=0.003):
    X = [[1, xi] for xi in X]  # 补上偏移量x0
    N = len(X[0])  # X的维度
    M = len(X)  # 样本个数
    theta = np.zeros((N,))  # 参数初始值
    last_theta = np.zeros(theta.shape)

    times = 10000
    while times > 0:
        times -= 1
        for i in range(M):
            last_theta[:] = theta[:]
            for j in range(N):
                theta[j] -= alpha * (np.dot(theta, X[i]) - Y[i]) * X[i][j]
        if np.sum((theta - last_theta) ** 2) <= epslion:  # 当前后参数的变化小于一定程度时可以终止迭代
            break
    return theta


# 根据输入值：X向量，即拟合阶数，计算对应的范德蒙矩阵
def vandermonde_matrix(X, Y, order=1):
    # 根据数据点构造X，Y的 范德蒙德矩阵
    m = len(Y)
    matX = np.array([[np.sum([X[i] ** (k2 + k1) for i in range(m)])
                      for k2 in range(order + 1)] for k1 in range(order + 1)])
    matY = np.array([np.sum([(X[i] ** k) * Y[i] for i in range(m)])
                     for k in range(order + 1)])
    theta = np.linalg.solve(matX, matY)
    return theta


if __name__ == "__main__":
    X, Y = randData()
    # theta = vandermonde_matrix(X, Y, order=1)
    theta = LS_line(X, Y)
    # theta = LS_sgd(X,Y)

    # 画出数据点与拟合曲线
    plt.figure()
    plt.plot(X, Y, linestyle='', marker='.')
    yhyp = [hypfunc(X[i], theta) for i in range(len(X))]
    plt.plot(X, yhyp, linestyle='-')
    plt.show()
