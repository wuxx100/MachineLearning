#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == '__main__':
    path='iris.data'
    data=pd.read_csv(path,header=None)
    x_prime=data[range(4)]
    y=pd.Categorical(data[4]).codes

    feature_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

    # for i,pair in enumerate(feature_pairs):
    pair=feature_pairs[0]
    x = x_prime[pair]
    RanForestModel = RandomForestClassifier(n_estimators=200,criterion='entropy',max_depth=3)
    RanForestModel.fit(x,y.ravel())

    #预测
    y_hat = RanForestModel.predict(x)
    print u'准确率', np.mean(y_hat==y.ravel())

    #显示
    x1_min, x2_min = x.min()
    x1_max, x2_max = x.max()
    x1_num = 50
    x2_num = 50

    t1=np.linspace(x1_min,x1_max,x1_num)
    t2 = np.linspace(x2_min, x2_max, x2_num)
    x1,x2=np.meshgrid(t1,t2)
    x_show=np.stack((x1.flat,x2.flat),axis=1)
    y_show_hat = RanForestModel.predict(x_show)
    y_show_hat=y_show_hat.reshape(x1.shape)


    cm_light=mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

    plt.pcolormesh(x1,x2,y_show_hat,cmap=cm_light)
    plt.scatter(x[pair[0]],x[pair[1]],c=y.ravel(),s=40,cmap=cm_dark)
    plt.show()
