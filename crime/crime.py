#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/6/30 13:58
#@Author: czj
#@File  : crime.py

import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# train_path = 'D:\\MyInstallData\\PyCharm\\Kaggle\\crime\\train.csv'
# test_path = 'D:\\MyInstallData\\PyCharm\\Kaggle\\crime\\test.csv'
train_path = './train.csv'
test_path = './test.csv'
train = pd.read_csv(train_path,sep=',')
test = pd.read_csv(train_path,sep=',')

#categroy labels(39个)
#'Dates', 'Category', 'Descript', 'DayOfWeek',
#'PdDistrict','Resolution', 'Address', 'X', 'Y'
# train.Category.unique()
# train.Category.nunique()
# from sklearn import preprocessing
# enc = preprocessing.OrdinalEncoder()
# enc.fit_transform(train['Category'])
# sns.relplot(kind='scatter',data=train,x='X',y='Y',hue='Category')

targets = ['WARRANTS', 'OTHER OFFENSES', 'LARCENY/THEFT', 'VEHICLE THEFT',
       'VANDALISM', 'NON-CRIMINAL', 'ROBBERY', 'ASSAULT', 'WEAPON LAWS',
       'BURGLARY', 'SUSPICIOUS OCC', 'DRUNKENNESS',
       'FORGERY/COUNTERFEITING', 'DRUG/NARCOTIC', 'STOLEN PROPERTY',
       'SECONDARY CODES', 'TRESPASS', 'MISSING PERSON', 'FRAUD',
       'KIDNAPPING', 'RUNAWAY', 'DRIVING UNDER THE INFLUENCE',
       'SEX OFFENSES FORCIBLE', 'PROSTITUTION', 'DISORDERLY CONDUCT',
       'ARSON', 'FAMILY OFFENSES', 'LIQUOR LAWS', 'BRIBERY',
       'EMBEZZLEMENT', 'SUICIDE', 'LOITERING',
       'SEX OFFENSES NON FORCIBLE', 'EXTORTION', 'GAMBLING', 'BAD CHECKS',
       'TREA', 'RECOVERED VEHICLE', 'PORNOGRAPHY/OBSCENE MAT']
weekofDay = {
        'Wednesday':2,'Tuesday':1, 'Monday':0, 'Sunday':6, 'Saturday':6, 'Friday':4,'Thursday':3
    }
districts = {c:i for i,c in enumerate(train['PdDistrict'].unique())}
def feature_engineer(data):
    data['DayOfWeek'] = data.DayOfWeek.apply(lambda x:weekofDay[x])
    data['PdDistrict'] = data.PdDistrict.apply(lambda x:districts[x])
    dateIdx = pd.DatetimeIndex(data.Dates)
    data['Month'] = dateIdx.month
    data['Hour'] = dateIdx.hour
    data['Day'] = dateIdx.day
    data['Category'] = data.Category.apply(lambda x:targets.index(x))
    return data

X_all = ['Month','Hour','Day','PdDistrict','X','Y','DayOfWeek']
train = feature_engineer(train)
x_train,y_train = train.loc[:,X_all], train['Category']

test = feature_engineer(test)
x_test,y_test = test.loc[:,X_all], test['Category']


# 随机森林
def randomForest():
    print("randomForest start....")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    params = {
        'max_features':['sqrt','log2'],
    }
    rfc = RandomForestClassifier(n_estimators=100,n_jobs=30)
    gs = GridSearchCV(estimator=rfc,param_grid=params, cv=10)
    gs.fit(x_train, y_train)
    print('gs.best_score_:',gs.best_score_)
    print('gs.best_params_:',gs.best_params_)
    print('gs.best_estimator_:',gs.best_estimator_)
    print('train acc:', gs.score(x_train, y_train))
    print('test acc:', gs.score(x_test, y_test))
def KNN():
    print("KNN start....")
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    knn = KNeighborsClassifier(n_neighbors=4,weights = 'uniform')
    params = {
        'n_neighbors':np.arange(50,800,50)
    }
    gs = GridSearchCV(estimator=knn, param_grid=params, cv=10)
    gs.fit(x_train,y_train)
    print('gs.best_score_:', gs.best_score_)
    print('gs.best_params_:', gs.best_params_)
    print('gs.best_estimator_:', gs.best_estimator_)
    print('train acc:', gs.score(x_train, y_train))
    print('test acc:', gs.score(x_test, y_test))

if __name__=='__main__':
    # randomForest()
    KNN()

##############思考
#1. 多分类标签中 某一个类别中出现的比较多