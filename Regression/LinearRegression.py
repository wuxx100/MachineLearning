#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


if __name__ == '__main__':
    # 读入广告销量影响数据
    data = pd.read_csv('Advertising.csv')
    x=data[['TV','Radio','Newspaper']]
    y=data['Sales']
    print x.shape
    print y.shape

    # 取训练集与测试集
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8, random_state=1)

    # 选取线性回归模型(MLE)处理训练集得到模型(LRmodel)
    LRmodel=LinearRegression()
    LRmodel.fit(x_train,y_train)
    print LRmodel.coef_, LRmodel.intercept_ # 参数与截距

    # 用模型验证测试集


