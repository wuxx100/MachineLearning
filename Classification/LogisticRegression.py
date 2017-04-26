#!/usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
import matplotlib as mpl
import  matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if __name__ == '__main__':
    file='iris.data'
    data=pd.read_csv(file, header=None)         #文件第一行没有类别
    # print data[4]
    data[4]=pd.Categorical(data[4]).codes
    # print data[4]

    x,y=np.split(data.values, (4,),axis=1)

    x=x[:, 2:4]

    LogisRModel=Pipeline([
        ('sc',StandardScaler()),                  #将数据标准化
        ('poly',PolynomialFeatures(degree=2)),
        ('clf',LogisticRegression())              #classification filter分类器
    ])

    LogisRModel.fit(x,y.ravel())
    y_hat=LogisRModel.predict(x)

    #y_hat_prob=LogisRModel.predict_proba(x)
    print u'准确度:', (np.mean(y_hat==y.ravel()))


    ## 画图
    x1_min = x[:, 0].min()
    x1_max = x[:, 0].max()
    x2_min = x[:, 1].min()
    x2_max = x[:, 1].max()
    x1_num=500
    x2_num=500
    t1=np.linspace(x1_min,x1_max,x1_num)
    t2 = np.linspace(x2_min, x2_max, x2_num)
    x1,x2=np.meshgrid(t1,t2)
    x_test=np.stack((x1.ravel(),x2.ravel()),axis=1)             #把数据放平才可以预测
    y_hat = LogisRModel.predict(x_test)
    y_hat=y_hat.reshape(x1.shape)                                     #把数据恢复尺寸


    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])     #规定背景颜色
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

    plt.figure()
    plt.pcolormesh(x1,x2,y_hat,cmap=cm_light)
    plt.scatter(x[:,0],x[:,1],c=y,cmap=cm_dark)
    plt.xlabel(u'花瓣长度', fontsize=14)
    plt.ylabel(u'花瓣宽度', fontsize=14)
    plt.grid()
    patchs = [mpatches.Patch(color='#77E0A0', label='Iris-setosa'),
              mpatches.Patch(color='#FF8080', label='Iris-versicolor'),
              mpatches.Patch(color='#A0A0FF', label='Iris-virginica')]
    plt.legend(handles=patchs)
    plt.title(u'鸢尾花Logistic回归')
    plt.show()




