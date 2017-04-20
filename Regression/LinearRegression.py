#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import RidgeCV, LassoCV
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    # 读入广告销量影响数据
    data = pd.read_csv('Advertising.csv')
    x=data[['TV','Radio']]
    y=data['Sales']
    # print x.shape
    # print y.shape

    # 取训练集与测试集
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8, random_state=1)
    order = y_test.argsort()  # 将测试数据排序，从小到大，为了使数据美观
    y_test = y_test.values[order]  # 将y从Series转换到array
    x_test = x_test.values[order, :]

    ###################################################################################################
    # 选取线性回归模型(MLE)处理训练集得到模型(LRmodel)
    print '############################################################################################'
    print 'LR:'
    LRmodel=LinearRegression()
    LRmodel.fit(x_train,y_train)
    print LRmodel.coef_, LRmodel.intercept_ # 参数与截距

    # 用模型验证测试集
    y_hat_LR = LRmodel.predict(x_test)         # 预测
    mse = np.average((y_test-y_hat_LR)**2)
    rmse= np.sqrt(mse)

    print 'mse=',mse
    print 'rmse=',rmse
    print 'R2_train=', LRmodel.score(x_train,y_train)     # R^2=1-(RSS残差平方和)/(TSS总平方差和)
    print 'R2_test=', LRmodel.score(x_test,y_test)

    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(20,9))
    plt.subplot(131)
    t=np.arange(len(x_test))
    plt.plot(t,y_test,'r*',label=u'真实数据')
    plt.plot(t, y_hat_LR, 'g-', label=u'预测数据')
    plt.legend(loc='upper right')
    plt.title(u'线性回归销量预测')
    plt.grid(b=True)
    ###################################################################################################

    ###################################################################################################
    # 选取Ridge线性回归（L2-norm）
    print '############################################################################################'
    print 'Ridge:'
    # 先给定参数alpha
    alpha=np.logspace(-2,5,10)


    Ridgemodel=Ridge()
    Ridgemodel=GridSearchCV(Ridgemodel,param_grid={'alpha':alpha},cv=5)     # 取五折交叉验证
    Ridgemodel.fit(x_train,y_train)
    print Ridgemodel.best_params_

    y_hat_Ridge=Ridgemodel.predict(x_test)
    mse = np.average((y_test - y_hat_Ridge) ** 2)
    rmse = np.sqrt(mse)

    print 'mse=', mse
    print 'rmse=', rmse
    print 'R2_train=', Ridgemodel.score(x_train, y_train)  # R^2=1-(RSS残差平方和)/(TSS总平方差和)
    print 'R2_test=', Ridgemodel.score(x_test, y_test)

    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    plt.subplot(132)
    plt.plot(t,y_hat_Ridge,'g-',label=u'预测值')
    plt.plot(t,y_test,'r*',label=u'真实值')
    plt.legend(loc='upper right')
    plt.title(u'RIDGE 线性回归')
    plt.grid()
    ###################################################################################################

    ###################################################################################################
    # 选取Lasso线性回归（L1-norm）
    print '############################################################################################'
    print 'Lasso:'

    Lassomodel = Lasso()
    Lassomodel = GridSearchCV(Lassomodel, {'alpha':alpha}, cv=5)
    Lassomodel.fit(x_train, y_train)
    print Lassomodel.best_params_

    y_hat_Lasso = Lassomodel.predict(x_test)
    mse = np.average((y_test - y_hat_Lasso) ** 2)
    rmse = np.sqrt(mse)

    print 'mse=', mse
    print 'rmse=', rmse
    print 'R2_train=', Lassomodel.score(x_train, y_train)  # R^2=1-(RSS残差平方和)/(TSS总平方差和)
    print 'R2_test=', Lassomodel.score(x_test, y_test)

    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False


    plt.subplot(133)
    plt.plot(t, y_hat_Lasso, 'g-', label=u'预测值')
    plt.plot(t, y_test, 'r*', label=u'真实值')
    plt.legend(loc='upper right')
    plt.title(u'Lasso 线性回归')
    plt.grid()
    plt.show()
    ###################################################################################################

    ###################################################################################################
    # 选取RidgeCV线性回归（L2-norm）
    print '############################################################################################'
    print 'RidgeCV:'

    RidgeCVmodel = RidgeCV(alphas=alpha)
    RidgeCVmodel.fit(x_train, y_train)
    print RidgeCVmodel.coef_, RidgeCVmodel.intercept_, RidgeCVmodel.alpha_

    y_hat_RidgeCV = RidgeCVmodel.predict(x_test)
    mse = np.average((y_test - y_hat_RidgeCV) ** 2)
    rmse = np.sqrt(mse)

    print 'mse=', mse
    print 'rmse=', rmse
    print 'R2_train=', RidgeCVmodel.score(x_train, y_train)  # R^2=1-(RSS残差平方和)/(TSS总平方差和)
    print 'R2_test=', RidgeCVmodel.score(x_test, y_test)

    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(15, 9))
    plt.subplot(121)
    plt.plot(t, y_hat_RidgeCV, 'g-', label=u'预测值')
    plt.plot(t, y_test, 'r*', label=u'真实值')
    plt.legend(loc='upper right')
    plt.title(u'RidgeCV 线性回归')
    plt.grid()
    ###################################################################################################

    ###################################################################################################
    # 选取LassoCV线性回归（L1-norm）
    print '############################################################################################'
    print 'LassoCV:'

    LassoCVmodel = LassoCV(alphas=alpha)
    LassoCVmodel.fit(x_train,y_train)
    print LassoCVmodel.coef_.ravel(), LassoCVmodel.intercept_,LassoCVmodel.alpha_

    y_hat_LassoCV = LassoCVmodel.predict(x_test)
    mse = np.average((y_test - y_hat_LassoCV) ** 2)
    rmse = np.sqrt(mse)

    print 'mse=', mse
    print 'rmse=', rmse
    print 'R2_train=', LassoCVmodel.score(x_train, y_train)  # R^2=1-(RSS残差平方和)/(TSS总平方差和)
    print 'R2_test=', LassoCVmodel.score(x_test, y_test)

    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    plt.subplot(122)
    plt.plot(t, y_hat_LassoCV, 'g-', label=u'预测值')
    plt.plot(t, y_test, 'r*', label=u'真实值')
    plt.legend(loc='upper right')
    plt.title(u'LassoCV 线性回归')
    plt.grid()
    plt.show()







