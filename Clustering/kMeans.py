# !usr/bin/python
# -*- coding:utf-8 -*-

##注意！！！！！！！！！这里没有解释准确度的概念，之后可能加上

import sklearn.datasets as ds
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import colors

if __name__ == '__main__':

    N=400
    centersNum=4
    data, y = ds.make_blobs(N, centers=centersNum,cluster_std=(1,1.5,1,1.2),n_features=2)        ##n_features是样本的维度 ,一旦加入 cluster_std=(1,2,0.5,2)，
                                                                        # 导致方差不相等，结果就很差
    data2 = np.vstack((data[y==0][:],data[y==1][:50],data[y==2][:10],data[y==3][:5]))               ##不均衡样本
    # print data2

    kMeansModel=KMeans(n_clusters=4, init='k-means++')                                              #k-means++ 采用距离原点距离远近为概率选取下一个关键点
    y_hat=kMeansModel.fit_predict(data)
    y_hat2=kMeansModel.fit_predict(data2)

    R=np.array(((0,1),(1,0)))
    dataR=data.dot(R)
    y_hatR=kMeansModel.fit_predict(dataR)

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    cm=mpl.colors.ListedColormap(list('rgbm'))

    plt.figure()
    plt.subplot(221)
    plt.title(u'原始数据')
    plt.scatter(data[:,0], data[:,1], c=y, s=40)
    plt.grid(True)

    plt.subplot(222)
    plt.title(u'预测数据')
    plt.scatter(data[:, 0], data[:, 1], c=y_hat, s=40)
    plt.grid(True)

    plt.subplot(223)
    plt.title(u'旋转数据')
    plt.scatter(dataR[:, 0], dataR[:, 1], c=y_hatR, s=40)
    plt.grid(True)

    plt.subplot(224)
    plt.title(u'不均衡数据')
    plt.scatter(data2[:, 0], data2[:, 1], c=y_hat2, s=40)
    plt.grid(True)

    plt.show()