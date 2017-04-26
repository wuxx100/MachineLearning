#!/usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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