# -*- coding: utf-8 -*-

import numpy as np  #惯例
import scipy as sp  #惯例
from scipy.optimize import leastsq #这里就是我们要使用的最小二乘的函数
import pylab as pl

    
m = 9  #多项式的次数
 
def real_func(x):
    return np.sin(2*np.pi*x) #sin(2 pi x)
 
def fake_func(p, x):
    f = np.poly1d(p) #多项式分布的函数
    return f(x)
regularization = 0.00005
def residuals(p, y, x):
    ret = y - fake_func(p, x)
    ret = np.append(ret, np.sqrt(regularization)*p) #将lambda^(1/2)p加在了返回的array的后面
    return ret
'''    
#残差函数
def residuals(p, y, x):
    return y - fake_func(p, x)
'''
#随机选了9个点，作为x
x = np.linspace(0, 1, 12)
#画图的时候需要的“连续”的很多个点
x_show = np.linspace(0, 1, 1000)
 
y0 = real_func(x)
#加入正态分布噪音后的y
y1 = [np.random.normal(0, 0.1) + y for y in y0]
 
#先随机产生一组多项式分布的参数
p0 = np.random.randn(m)
 
plsq = leastsq(residuals, p0, args=(y1, x))
 
print 'Fitting Parameters ：', plsq[0] #输出拟合参数
 
pl.plot(x_show, real_func(x_show), label='real')
pl.plot(x_show, fake_func(plsq[0], x_show), label='fitted curve')
pl.plot(x, y1, 'bo', label='with noise')
pl.legend()
pl.show()