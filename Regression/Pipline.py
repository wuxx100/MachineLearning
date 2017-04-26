#!/usr/bin/python
# -*- coding:utf-8 -*-

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.exceptions import ConvergenceWarning
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == '__main__':
    warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
    np.random.seed(0)
    N=9
    x=np.linspace(0,6,N)+np.random.randn(N)
    x=np.sort(x)
    y = x ** 2 - 4 * x - 3 + np.random.randn(N)
    x_hat = np.linspace(x.min(), x.max(), 100)
    x_hat.shape=-1,1
    x.shape=-1,1

    models=[Pipeline([('poly',PolynomialFeatures()),
                      ('linear',LinearRegression())]),
            Pipeline([('poly',PolynomialFeatures()),
                      ('linear',RidgeCV(alphas=np.logspace(-2,2,10)))]),
            Pipeline([('poly',PolynomialFeatures()),
                      ('linear',LassoCV(alphas=np.logspace(-2,2,10)))]),
            ]

    dimention=np.arange(1,6) #阶
    clrs=[] #颜色
    for c in np.linspace(16711680, 255, dimention.size):
        clrs.append('#%06x' % c)

    line_width=np.linspace(5,2, dimention.size)
    titles=u'线性回归',u'Ridge回归',u'LASSO回归'

    plt.figure(figsize=(20,7))
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    for t in range(3):
        model=models[t]
        plt.subplot(1,3,t+1)
        plt.plot(x,y,'ro',ms=10)
        for d in dimention:
            model.set_params(poly__degree=d)
            model.fit(x,y)
            lin=model.get_params('linear')['linear']


            y_hat=model.predict(x_hat.reshape(-1,1))        #x按数列排，y按行排
            s_train=model.score(x,y)

            label=u'%d阶,$R^2$=%.3f' %(d,s_train)
            if hasattr(lin,'alpha_'):
                alp=lin.alpha_
                label=label+(' alpha=%.3f' %alp)
            print label
            z = N - 1 if (d == 2) else 0        #把二阶画在最上边
            plt.plot(x_hat,y_hat, color=clrs[d-1],lw=line_width[d-1],label=label,zorder=z)
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.title(titles[t])

    plt.show()

