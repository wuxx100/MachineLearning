# !/usr/bin/python
# -*- coding:utf-8 -*-

## 谱聚类
import numpy as np
import sklearn.datasets as ds
from sklearn.metrics import euclidean_distances
from sklearn.cluster import spectral_clustering
from sklearn.cluster import  SpectralClustering
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == '__main__':
    np.random.seed(0)
    N = 400
    data1, y1 = ds.make_blobs(N, 2, centers=((-1, 1), (1, 1), (1, -1), (-1, -1)), cluster_std=(0.5, 0.2, 0.2, 0.4))
    data1 = np.array(data1)
    # noiseNum = int(0.1 * N)
    # noise_01 = np.random.rand(noiseNum, 2)  ##行列，(0,1)之间

    # dataMin_1, dataMin_2 = np.min(data1, axis=0)
    # dataMax_1, dataMax_2 = np.max(data1, axis=0)

    # noise = np.zeros((noiseNum, 2))
    # noise[:, 0] = noise_01[:, 0] * (dataMax_1 - dataMin_1) + dataMin_1
    # noise[:, 1] = noise_01[:, 1] * (dataMax_2 - dataMin_2) + dataMin_2
    #
    # data1 = np.concatenate((data1, noise), axis=0)
    # yWithNoise = np.concatenate((y1, [4] * noiseNum))

    t=np.arange(0,2*np.pi,0.1)
    circle1=np.vstack((np.cos(t),np.sin(t))).T
    circle2 = np.vstack((2*np.cos(t), 3*np.sin(t))).T
    circle3 = np.vstack((3*np.cos(t), 4*np.sin(t))).T
    circle4 = np.vstack((5 * np.cos(t), 5 * np.sin(t))).T
    data2=np.vstack((circle1,circle2,circle3,circle4))



    for i, data in enumerate((data1,data2)):
    # for i in range(1):
    #     data=data2
        clustersNum=4
        cm = mpl.colors.ListedColormap(['r', 'g', 'b', 'm', 'c'])

        euclideanDis2=euclidean_distances(data,squared=True)
        #注意，这里使用的是spectral_clustering，不是SpectralClustering，所以就是个函数，不是个类

        for i,s in enumerate(np.logspace(-2,2,30)):
            print s
            affinity=np.exp(-euclideanDis2/(s**2)) + 1e-6   ##谱聚类的连通性，使用w_ij=exp(-|...|/2s^2)，应该加了正则项
            y_hat=spectral_clustering(affinity=affinity,n_clusters=clustersNum,assign_labels='kmeans')

            # print y_hat
            plt.figure()
            plt.scatter(data[:,0],data[:,1],c=y_hat,cmap=cm)
            plt.grid(True)
            plt.show()


## 对于data1,s=1.0表现最好
## 对于data2,s=0.33表现最好
