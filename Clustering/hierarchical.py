# !/usr/bin/python
# -*- coding:utf-8 -*-

##层次聚类
##http://blog.sina.com.cn/s/blog_7103b28a0102w4e1.html

import numpy as np
import sklearn.datasets as ds
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

if __name__ == '__main__':

    ##构造数据
    np.random.seed(0)
    N=400
    data,y = ds.make_blobs(N,2,centers=((-1,1),(1,1),(1,-1),(-1,-1)),cluster_std=(0.5,0.2,0.2,0.4))
    data=np.array(data)
    noiseNum=int(0.1*N)
    noise_01=np.random.rand(noiseNum,2)        ##行列，(0,1)之间

    dataMin_1,dataMin_2 = np.min(data,axis=0)
    dataMax_1, dataMax_2 = np.max(data,axis=0)

    noise=np.zeros((noiseNum,2))
    noise[:,0]=noise_01[:,0]*(dataMax_1-dataMin_1)+dataMin_1
    noise[:, 1] = noise_01[:, 1] * (dataMax_2 - dataMin_2) + dataMin_2

    dataWithNoise= np.concatenate((data,noise),axis=0)
    yWithNoise=np.concatenate((y, [4]*noiseNum))

    ## 使用vstack 和 hstack 效果一样
    # dataWithNoise2=np.vstack((data,noise))
    # yWithNoise2=np.hstack((y,[4]*noiseNum))


    ##使用融合聚类处理数据

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm = mpl.colors.ListedColormap(['r', 'g', 'b', 'm', 'c'])

    for i, (dataNow,yNow) in enumerate (((data,y),(dataWithNoise,yWithNoise))):
        clustersNum = 4
        neighborNum=7
        #设置连通性约束,设置只计算与周围neighborNum个点的距离
        connectivity=kneighbors_graph(dataNow,n_neighbors=neighborNum,include_self=True)
        clusterModel=AgglomerativeClustering(n_clusters=clustersNum, affinity='euclidean',linkage='ward',connectivity=connectivity)
        #采用欧拉距离，并且采用平均距离分类，ward是最小距离，complete是最大距离，average是平均距离
        #本数据ward表现最好，因为此时数据不是链状的

        y_hat=clusterModel.fit_predict(dataNow)
        plt.figure()
        plt.subplot(121)
        plt.scatter(dataNow[:,0],dataNow[:,1],c=yNow,cmap=cm)
        plt.grid(True)
        plt.subplot(122)
        plt.scatter(dataNow[:,0],dataNow[:,1],c=y_hat,cmap=cm)

        plt.show()
