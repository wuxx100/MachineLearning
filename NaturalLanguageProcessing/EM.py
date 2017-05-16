# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture     #这里假设数据是高斯混合模型(EM的一个实例)
from sklearn.metrics.pairwise import pairwise_distances_argmin      #用于计算聚类结果与原类别的顺序
import matplotlib as mpl

if __name__ == '__main__':
    path='iris.data'
    data = pd.read_csv(path, header=None)
    featurePrime, y = data[range(4)],data[4]
    y=pd.Categorical(y).codes

    componentNum=3
    feature_pair=[[0,1],[2,3]]

    for i, pair in enumerate(feature_pair):
        data=featurePrime[pair]
        ##对于np.array(由np.split(data.values, (4,),axis=1)来)，可以直接使用x=x[:, 2:4]，但是对于pd.DataFrame,只能使用x=x[[2,3]]
        ##而且对于np.array来说，分开之后再取用就是从x[0]开始，但是对于DataFrame，取用得用x[2],x[3](因为是DataFrame，列属性一直跟着)
        mean=np.array([np.mean(data[y==i]) for i in range(3)])       #实际均值
        cov=np.array([np.cov(data[y==i]) for i in range(3)])       #实际均值

        print '实际均值为:', mean
        print '实际cov:', cov

        gmmModel=GaussianMixture(n_components=componentNum,covariance_type='full')
        ##covariance_type可以是spherical, diag,tied, full分别例如[1,0;0,1],[1,0;0,2],[1,2;2,4],[1,2;3,4]这几种
        gmmModel.fit(data)
        print '预测均值:', gmmModel.means_
        print '预测方差:', gmmModel.covariances_
        y_hat=gmmModel.predict(data)

        order=pairwise_distances_argmin(mean, gmmModel.means_, metric='euclidean')  #看预测的mean在原始的mean中顺序如何
        print 'order:', order

        ##计算准确率
        for i in range(componentNum):
            y_hat[y_hat==i]=order[i]+componentNum
        y_hat-=componentNum
        #在这里倒一下顺序
        acc=np.mean(y_hat==y)
        print '准确率:', acc

        cm_light = mpl.colors.ListedColormap(['#FF8080', '#77E0A0', '#A0A0FF'])
        cm_dark = mpl.colors.ListedColormap(['r', 'g', '#6060FF'])

        x1_min, x2_min = data.min()
        x1_max, x2_max = data.max()

        x1_num = 500
        x2_num = 500
        t1 = np.linspace(x1_min, x1_max, x1_num)
        t2 = np.linspace(x2_min, x2_max, x2_num)
        x1, x2 = np.meshgrid(t1, t2)
        x_test = np.stack((x1.ravel(), x2.ravel()), axis=1)  # 把数据放平才可以预测
        y_hat_mesh = gmmModel.predict(x_test)

        for i in range(componentNum):
            y_hat_mesh[y_hat_mesh==i]=order[i]+componentNum
        y_hat_mesh-=componentNum

        y_hat_mesh = y_hat_mesh.reshape(x1.shape)

        plt.figure()
        plt.pcolormesh(x1,x2,y_hat_mesh,cmap=cm_light)
        plt.scatter(data[pair[0]],data[pair[1]],c=y,cmap=cm_dark)       ##这里用的是真实类别
        plt.grid(True)
        plt.show()