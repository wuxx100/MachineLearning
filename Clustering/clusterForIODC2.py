# !usr/bin/python
# -*- coding:utf-8 -*-

## IODC是之后给数据分类的重要依据，但是本身不是鲁棒的，所以需要通过其他几个鲁棒的数据先估计出正确值
## 这里准备先用DBSCAN，因为并不知道会有几个IODC值

#密度聚类DBSCAN
import sklearn.datasets as ds
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

if __name__ == '__main__':
    prnNum=14
    path = 'findIODC/prn'+str(prnNum)+'.csv'
    data=pd.read_csv(path,header=None)
    ##这里用m0(13),deltaN(12),cis(21),toe(18),cic(19)  都试试
    featurePrime=np.array(data)
    feature=featurePrime[:,range(7,33)]
    feature = StandardScaler().fit_transform(feature)

    for i in range(1):
        # print a, b
        feature1=feature[:,[13-7,12-7,19-7]]
        # feature2=featurePrime[:,21]

        DBSCANModel = DBSCAN(eps=0.1, min_samples=5)

        y_hat=DBSCANModel.fit_predict(feature1)

        core_indices = np.zeros(y_hat.shape, dtype=bool)  ##初始化核心点
        core_indices[DBSCANModel.core_sample_indices_] = True  # 设核心点

        yUnique = np.unique(y_hat)
        clusterNum = yUnique.size - (1 if -1 in y_hat else 0)  ##分类个数，在密度聚类里，这个不是超参数，是结果
        print clusterNum
        ##分类是-1 代表是噪声

        clrs = plt.cm.Spectral(np.linspace(0, 0.8, yUnique.size))  ##由于不知道分成多少组，无法事先给颜色，就这样用Spectral给
        plt.figure()

        for k, color in zip(yUnique, clrs):
            curIdx = (y_hat == k)  # 当前等于y_hat==k的序号，此时是同一个颜色（此时的color）
            sizeForGroup=curIdx.tolist().count(True)
            if k == -1:  # 噪声
                plt.scatter(feature1[curIdx, 0], feature1[curIdx, 1], s=20, c='k')
                print 'for group', k
                print featurePrime[curIdx, 33]
                continue
            plt.scatter(feature1[curIdx, 0], feature1[curIdx, 1], s=30, c=color, edgecolors='k')
            plt.scatter(feature1[curIdx & core_indices][:, 0], feature1[curIdx & core_indices][:, 1], s=sizeForGroup*10, c=color, marker='o',
                        edgecolors='k')
            print 'for group', k
            print np.unique(featurePrime[curIdx,33])
            for i in np.unique(featurePrime[curIdx,33]):
                data[data[33]==i].to_csv('findIODC/cleanedData/prn_'+str(prnNum)+'_IODC_'+str(int(i))+'_.csv',header=None,index=True)


        plt.grid(True)

        plt.show()
