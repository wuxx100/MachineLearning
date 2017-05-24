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
import os
import string

if __name__ == '__main__':
    year=13
    for day in range(2,3):
        readFolder = 'navData/'+str(year)+'/'+str(day)+'/findIODC/cleanedData/'
        files = os.listdir(readFolder)
        for fileName in files:
            if string.find(fileName, '.csv') != -1:
                splitName=fileName.split('_')
                print 'For PRN=',splitName[1],'IODC=', splitName[3]

                data=pd.read_csv('navData/'+str(year)+'/'+str(day)+'/findIODC/cleanedData/'+fileName,header=None)
                data[35]=data[35]%604800

                featurePrime=np.array(data)

                feature=featurePrime[:,range(8,36)]
                feature = StandardScaler().fit_transform(feature)
                ##TTOM(34),URA(30)
                feature1=feature[:,[35-8,31-8]]
                DBSCANModel = DBSCAN(eps=0.005, min_samples=2)
                y_hat=DBSCANModel.fit_predict(feature1)

                core_indices = np.zeros(y_hat.shape, dtype=bool)  ##初始化核心点
                core_indices[DBSCANModel.core_sample_indices_] = True  # 设核心点

                print 'group for each point is: ',y_hat
                yUnique = np.unique(y_hat)
                clusterNum = yUnique.size - (1 if -1 in y_hat else 0)  ##分类个数，在密度聚类里，这个不是超参数，是结果
                print 'group number is(without noise):',clusterNum
                ##分类是-1 代表是噪声
                clrs = plt.cm.Spectral(np.linspace(0, 0.8, yUnique.size))  ##由于不知道分成多少组，无法事先给颜色，就这样用Spectral给
                # plt.figure()
                kForMinTTOM=-1
                minTTOM=604801
                lineNum=10000
                for k, color in zip(yUnique, clrs):
                    curIdx = (y_hat == k)  # 当前等于y_hat==k的序号，此时是同一个颜色（此时的color）
                    sizeForGroup=curIdx.tolist().count(True)
                    if k == -1:
                        # plt.scatter(feature1[curIdx, 0], feature1[curIdx, 1], s=20, c='k')
                        print 'for noise',k
                        print featurePrime[curIdx, 31],featurePrime[curIdx, 35]
                        continue
                    # plt.scatter(feature1[curIdx, 0], feature1[curIdx, 1], s=30, c=color, edgecolors='k')
                    # plt.scatter(feature1[curIdx & core_indices][:, 0], feature1[curIdx & core_indices][:, 1], s=sizeForGroup*10, c=color, marker='o',
                    #              edgecolors='k')
                    print 'for group', k
                    TTOM=min(np.unique(featurePrime[curIdx, 35]))
                    URA=np.unique(featurePrime[curIdx, 31])
                    print 'URA=',URA,'TTOM=',TTOM

                    if TTOM<minTTOM:
                        minTTOM=TTOM
                        kForMinTTOM=k
                        lineNum=min(featurePrime[curIdx,0])
                        print lineNum
                print 'kFOrMinTTOM=',kForMinTTOM
                resultData=data[data[0]==lineNum]
                print resultData[range(1, 39)]
                resultData[range(1, 39)].to_csv('navData/'+str(year)+'/'+str(day)+'/findIODC/Result/myNavMes.csv',mode='a',header=None,index=None)
                print
                # plt.grid(True)
            # plt.show()
