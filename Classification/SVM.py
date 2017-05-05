# /usr/bin/python
# -*- encoding:utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path='iris.data'
    data=pd.read_csv(path, header=None)
    x,y=data[range(4)],data[4]
    y=pd.Categorical(y).codes
    x=x[[0,1]]
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=1)

    SVMModel=svm.SVC(C=0.1, kernel='linear', decision_function_shape= 'ovr')    #!!!!!!!!!!!!!!!!!!

    SVMModel.fit(x_train,y_train.ravel())

    print "训练集精度", SVMModel.score(x_train,y_train)
    print "测试集精度", SVMModel.score(x_test,y_test)

    #画图

    x1_min, x2_min= x.min()
    x1_max, x2_max= x.max()
    x1_num=500
    x2_num=500
    t1=np.linspace(x1_min,x1_max,x1_num)
    t2=np.linspace(x2_min,x2_max,x2_num)
    x1,x2=np.meshgrid(t1,t2)
    x_show=np.stack((x1.flat,x2.flat),axis=1)
    y_show_hat=SVMModel.predict(x_show)
    y_show_hat=y_show_hat.reshape(x1.shape)

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

    plt.figure()
    plt.pcolormesh(x1,x2,y_show_hat,cmap=cm_light)
    plt.scatter(x[0],x[1],c=y,s=40,cmap=cm_dark)
    plt.title(u'鸢尾花SVM分类')
    plt.grid(True)
    plt.show()
