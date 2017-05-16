# !/usr/bin/python
# -*- coding:utf-8 -*-

#密度聚类DBSCAN
import sklearn.datasets as ds
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np


if __name__ == '__main__':
    N=1000
    centers = [[1, 2], [-1, -1], [1, -1], [-1, 1]]
    data, y = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=[0.5, 0.25, 0.7, 0.5], random_state=0)
    # data1 = StandardScaler().fit_transform(data)          ##使方差标准化
    # plt.figure()
    # plt.scatter(data[:,0],data[:,1])
    # plt.show()
    # plt.figure()
    # plt.scatter(data1[:, 0], data1[:, 1])
    # plt.show()

    params = ((0.2, 5), (0.2, 10), (0.2, 15), (0.3, 5), (0.3, 10), (0.3, 15))  ##eps & m (密度半径 及 周边最小点数)

    for i in range(6):
        eps, minSamples=params[i]
        DBSCANModel = DBSCAN(eps=eps, min_samples=minSamples)
        y_hat=DBSCANModel.fit_predict(data)

        core_indices=np.zeros(y_hat.shape, dtype=bool)             ##初始化核心点
        core_indices[DBSCANModel.core_sample_indices_] = True       #设核心点

        yUnique=np.unique(y_hat)
        clusterNum=yUnique.size-(1 if -1 in y_hat else 0)         ##分类个数，在密度聚类里，这个不是超参数，是结果
                                                                        ##分类是-1 代表是噪声

        clrs = plt.cm.Spectral(np.linspace(0,0.8,yUnique.size))               ##由于不知道分成多少组，无法事先给颜色，就这样用Spectral给
        plt.figure()

        for k, color in zip(yUnique, clrs):
            curIdx=(y_hat==k)                               #当前等于y_hat==k的序号，此时是同一个颜色（此时的color）
            if k == -1:                                        #噪声
                plt.scatter(data[curIdx, 0], data[curIdx, 1], s=20, c='k')
                continue
            plt.scatter(data[curIdx, 0], data[curIdx, 1], s=30, c=color, edgecolors='k')
            plt.scatter(data[curIdx & core_indices][:, 0], data[curIdx & core_indices][:, 1], s=60, c=color, marker='o', edgecolors='k')

        plt.grid(True)

        plt.show()

