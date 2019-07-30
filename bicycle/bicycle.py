#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/11 16:40
#@Author: czj
#@File  : bicycle.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.family'] = 'sans-serif'

path = 'D:\\MyInstallData\\PyCharm\\Kaggle\\bicycle\\bikeTrain.csv'
train = pd.read_csv(path, sep=',')
#没有缺失值
# train.info
"""
'datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count'
"""
# sns.stripplot('season','count',data=train)
# #holiday  有明确的关系
# sns.stripplot('holiday','count',data=train)
# sns.stripplot('workingday','count',data=train)

# dateIdx = pd.DatetimeIndex(train.datetime)
# train['month'] = dateIdx.month.values
# train['dayofweek'] = dateIdx.dayofweek.values
# train.loc[(train.dayofweek==5) | (train.dayofweek==6),'isWork'] = 0
# train.loc[train.isWork.isnull(), 'isWork'] = 1

train['month'] = pd.DatetimeIndex(train.datetime).month
train['day'] = pd.DatetimeIndex(train.datetime).dayofweek
train['hour'] = pd.DatetimeIndex(train.datetime).hour


#建立随机森林预测
# train score: 0.9816433869987461
# test score: 0.8637780495566116
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
rfr = RandomForestRegressor(n_estimators=100,verbose=3)
col = ['season', 'holiday', 'workingday', 'weather', 'temp','atemp', 'humidity', 'windspeed', 'month', 'day','hour']
x1,x2,y1,y2 = train_test_split(train[col].as_matrix(),train['count'].as_matrix(),test_size=0.2,random_state=0)
rfr.fit(x1,y1)
print("train score:",rfr.score(x1,y1))
print("test score:",rfr.score(x2,y2))

#岭回归
# train score: 0.3388603529147849
# test score: 0.33199721416235206
from sklearn.linear_model import Ridge
r = Ridge()
r.fit(x1,y1)
print("train score:",r.score(x1,y1))
print("test score:",r.score(x2,y2))

#SVM
# train score: 0.48200560658965014
# test score: 0.45215172520986213
from sklearn.svm import SVR
svr = SVR(kernel='rbf',C=10,gamma=0.01)
svr.fit(x1,y1)
print("train score:",svr.score(x1,y1))
print("test score:",svr.score(x2,y2))

#RandomForeset + GridSearch
from sklearn.model_selection import GridSearchCV
params = {
    "n_estimators":[10,100,500]
}
rfr = RandomForestRegressor()
gs = GridSearchCV(estimator=rfr,param_grid=params,scoring='r2',verbose=3)
gs.fit(x1,y1)
print("best estimator:",gs.best_estimator_)
print("best score:",gs.best_score_)
print("best params:",gs.best_params_)
print("test score:",gs.score(x2,y2))


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best")
        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()
    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff
plot_learning_curve(RandomForestRegressor(n_estimators=100),'Test',train[col].as_matrix(),train['count'].as_matrix(),cv=5)

#数据分析  观察不同 windspeed 下 count的次数
train.groupby('windspeed').mean().plot(y='count', marker='o')
train.groupby('humidity').mean().plot(y='count', marker='o')
train.groupby('temp').mean().plot(y='count', marker='o')

fig, axs = plt.subplots(2, 3, sharey=True)
train.plot(kind='scatter', x='temp', y='count', ax=axs[0, 0], figsize=(16, 8), color='magenta')
train.plot(kind='scatter', x='atemp', y='count', ax=axs[0, 1], color='cyan')
train.plot(kind='scatter', x='humidity', y='count', ax=axs[0, 2], color='red')
train.plot(kind='scatter', x='windspeed', y='count', ax=axs[1, 0], color='yellow')
train.plot(kind='scatter', x='month', y='count', ax=axs[1, 1], color='blue')
train.plot(kind='scatter', x='hour', y='count', ax=axs[1, 2], color='green')

# sns.pairplot(train[["temp", "humidity", "count"]], hue="count")
corr_col = ['temp','weather','windspeed','day', 'month', 'hour','count']
corr = train[corr_col].corr()
plt.figure()
plt.matshow(corr)
plt.colorbar()
plt.show()