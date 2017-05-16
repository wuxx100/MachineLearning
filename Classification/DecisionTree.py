#!/usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pydotplus
import matplotlib as mpl
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file = 'iris.data'
    data = pd.read_csv(file, header=None)  # 文件第一行没有类别
    data[4] = pd.Categorical(data[4]).codes

    x,y = np.split(data.values,(4,),axis=1)
    x=x[:,2:4]

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)

    DecTreeModel=DecisionTreeClassifier(criterion='entropy')    #以熵为判据
    DecTreeModel.fit(x_train,y_train)
    y_test_hat=DecTreeModel.predict(x_test)

    print u'准确度:', (np.mean(y_test_hat==y_test.ravel()))


    # 保存决策树为pdf

    # 花萼长度、花萼宽度，花瓣长度，花瓣宽度
    iris_feature_E = 'sepal length', 'sepal width', 'petal length', 'petal width'
    iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
    iris_class = 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'
    # with open('iris.dot','w') as f:
    #     tree.export_graphviz(DecTreeModel,out_file=f)
    # dot_data=tree.export_graphviz(DecTreeModel,out_file=None, feature_names=iris_feature_E, class_names=iris_class, filled=True, rounded=True)
    # graph=pydotplus.graph_from_dot_data(dot_data)
    # graph.write_pdf('iris.pdf')

    # 画图表示结果
    x1_min = x[:, 0].min()
    x1_max = x[:, 0].max()
    x2_min = x[:, 1].min()
    x2_max = x[:, 1].max()
    x1_num = 500
    x2_num = 500
    t1 = np.linspace(x1_min, x1_max, x1_num)
    t2 = np.linspace(x2_min, x2_max, x2_num)
    x1, x2 = np.meshgrid(t1, t2)
    x_show = np.stack((x1.ravel(), x2.ravel()), axis=1)  # 把数据放平才可以预测
    y_show_hat = DecTreeModel.predict(x_show)
    y_show_hat = y_show_hat.reshape(x1.shape)  # 把数据恢复尺寸

    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])  # 规定背景颜色
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

    plt.figure()
    plt.pcolormesh(x1,x2,y_show_hat,cmap=cm_light)

    print 'y=',x[:,0]

    plt.scatter(x[:,0],x[:,1],c=y.ravel(),s=40,cmap=cm_dark)
    plt.xlabel(iris_feature[0], fontsize=15)
    plt.ylabel(iris_feature[1], fontsize=15)
    plt.title(u'鸢尾花决策树')
    plt.show()


    # 过拟合
    depth=np.arange(1,15)
    x_m, y_m = np.split(data.values, (4,), axis=1)
    x_m = x_m[:, :2]
    x_train_m, x_test_m, y_train_m, y_test_m = train_test_split(x_m, y_m, train_size=0.7, random_state=1)

    list_err=[]

    for d in depth:
        clf=DecisionTreeClassifier(criterion='entropy', max_depth=d)            #设置最深深度，也可以设置最小簇个数等
        clf.fit(x_train_m,y_train_m)
        y_test_hat_m=clf.predict(x_test_m)
        err=1-np.mean(y_test_hat_m==y_test_m.ravel())
        # print d, '采用花萼错误率：', err
        list_err.append(err)

    plt.figure()
    plt.plot(depth,list_err,'ro',lw=40)
    plt.plot(depth,list_err)
    plt.xlabel(u'深度')
    plt.ylabel(u'错误率')
    plt.grid(True)
    plt.title(u'决策树深度与过拟合')
    plt.show()