#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib as mpl


if __name__ == '__main__':
    # 读入广告销量影响数据
    data = pd.read_csv('Advertising.csv')
    x=data[['TV','Radio','Newspaper']]
    y=data['Sales']
    # print x.shape
    # print y.shape

    # 取训练集与测试集
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8, random_state=1)

    # 选取线性回归模型(MLE)处理训练集得到模型(LRmodel)
    LRmodel=LinearRegression()
    LRmodel.fit(x_train,y_train)
    print LRmodel.coef_, LRmodel.intercept_ # 参数与截距

    # 用模型验证测试集
    print 'y_test=',y_test,y_test.shape
    order = y_test.argsort()                # 将测试数据排序，从小到大，为了使数据美观
    y_test=y_test.values[order]             # 将y从Series转换到array
    x_test=x_test.values[order,:]
    y_hat = LRmodel.predict(x_test)         # 预测
    mse = np.average((y_test-y_hat)**2)
    rmse= np.sqrt(mse)

    print 'mse=',mse
    print 'rmse=',rmse
    print 'R2_train=', LRmodel.score(x_train,y_train)     # R^2=1-(RSS残差平方和)/(TSS总平方差和)
    print 'R2_test=', LRmodel.score(x_test,y_test)

    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    plt.figure()
    t=np.arange(len(x_test))
    plt.plot(t,y_test,'r*',label=u'真实数据')
    plt.plot(t, y_hat, 'g-', label=u'预测数据')
    plt.legend(loc='upper right')
    plt.title(u'线性回归销量预测')
    plt.grid(b=True)
    plt.show()





