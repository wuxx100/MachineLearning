# !/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib as mpl
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def analyseImg(imageV):
    binNum=10                       ##把0到1分成几类进行直方图表示
    density, edges = np.histogramdd(imageV,bins=[binNum,binNum,binNum], range=[(0,1), (0,1), (0,1)])
    density /= density.max()
    x=y=z=np.arange(binNum)
    coord=np.meshgrid(x,y,z)

    fig=plt.figure()
    frequency3D=fig.add_subplot(111,projection='3d')
    frequency3D.scatter(coord[0],coord[1],coord[2],s=100*density)

    plt.figure()
    density=np.sort(density[density>0])[::-1]           ##[::-1]是设从开始到结束，步长为-1（翻转）
    t=np.arange(len(density))
    plt.plot(t,density,'r-',density,'go')
    plt.grid(True)

    plt.show()



def restore_image(centerColor, classNum, imgShape):
    row, col, dep=imgShape
    imageRes=np.empty((row,col,3))
    i=0
    for r in range(row):
        for c in range(col):
            imageRes[r,c,:]=centerColor[classNum[i]]
            i += 1
    return imageRes

if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    centerNum=60                                        ##聚成60种颜色

    image255=Image.open('IMG.jpg')
    image=np.array(image255).astype(np.float)/255           ##astype为了设为小数，否则全是零

    image=image[:,:,:3]
    imageV=image.reshape(-1,3)

    analyseImg(imageV)

    N=imageV.shape[0]       #总点数
    ranIdx=np.random.randint(0,N,size=1000)     ##选取1000个点做聚类
    imageSample=imageV[ranIdx]

    kMeansModel = KMeans(n_clusters=centerNum, init='k-means++')
    kMeansModel.fit(imageSample)                ##用1000点聚类
    res=kMeansModel.predict(imageV)                 ##把模型应用在全部点上

    # print '聚类结果：\n', res
    # print '聚类中心：\n', model.cluster_centers_

    plt.figure()
    plt.subplot(121)
    plt.imshow(image)

    plt.subplot(122)
    imageAfterCluster=restore_image(kMeansModel.cluster_centers_,res,image.shape)
    plt.imshow(imageAfterCluster)

    plt.show()

