#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/2 20:03
#@Author: czj
#@File  : titanic.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train_path = 'D:\\MyInstallData\\PyCharm\\Kaggle\\titanic\\train.csv'
test_path = 'D:\\MyInstallData\\PyCharm\\Kaggle\\titanic\\test.csv'
train = pd.read_csv(train_path,sep=',')
test = pd.read_csv(test_path,sep=',')


train['Source'] = 'train'
test['Source'] = 'test'
data = pd.concat([train,test],axis=0)

#查看是否有确实值  缺失情况 Age:177  Cabin:687
data.apply(lambda x:x.isnull().sum())
#因为Age对于存活很重要，所以保留 删除Cabin和Name,PassengerId
data.drop(['Name','Cabin','PassengerId'], axis=1, inplace=True)


#查看每个连续值的样本分布
continous = ['Pclass','Age','SibSp','Parch','Fare']
discrete = ['Sex','Ticket','Embarked']

for var in continous:
    print(var)
    print(train[var].value_counts())

for var in discrete:
    print(var)
    print(train[var].value_counts())
#Ticket 这一列的值特别的多，因此不考虑在内部
data.drop(['Ticket'], axis=1, inplace=True)

#将Sex,Embarked变成one-hot
# from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder()
embark = pd.get_dummies(data['Embarked'],prefix='embark')
t = pd.concat([data,embark],axis=1)
sex = pd.get_dummies(data['Sex'],prefix='sex')
t = pd.concat([t,sex],axis=1)
t.drop(['Embarked','Sex'],axis=1,inplace=True)

#t.apply(lambda x:x.isnull().sum())
#对于Age中的缺失值进行聚类
from sklearn.neighbors import KNeighborsRegressor
#Age 为 null的
t['Fare'].fillna(t.Fare.mean(),inplace=True)
testidx = np.where(pd.isnull(t.Age)==True)[0]
trainidx = np.where(pd.isnull(t.Age)==False)[0]
testAge = t.iloc[testidx,:]
trainAge = t.iloc[trainidx,:]
knr = KNeighborsRegressor(n_neighbors=5)
knr.fit(trainAge.drop(['Age','Source','Survived'],axis=1), trainAge.Age)
testAge.drop(['Age'],axis=1,inplace=True)
testAge['Age'] = knr.predict(testAge.drop(['Source','Survived'],axis=1))
knn_age = pd.concat([trainAge,testAge],axis=0)

############################建立逻辑回归函数   在训练集上的结果0.8058  kaggle:0.53588
from sklearn.linear_model import LogisticRegressionCV
lr = LogisticRegressionCV(cv=10,penalty='l2')
train, test = knn_age.iloc[knn_age.Source.values=='train',:], knn_age.iloc[knn_age.Source.values=='test',:]
test.drop(['Survived','Source'],axis=1,inplace=True)
train.drop(['Source'],axis=1,inplace=True)
lr.fit(train.drop(['Survived'],axis=1), train['Survived'])
lr.score(train.drop(['Survived'],axis=1), train['Survived'])

test_Survived = lr.predict(test).astype(np.int).reshape(-1,1)
id = np.arange(892,892+len(test_Survived),dtype=np.int).reshape(-1,1)
gender_submission = np.hstack((id,test_Survived))

np.savetxt(fname='./submission.csv', X=gender_submission,fmt='%d,%d',
               header='PassengerId,Survived', comments='')
############################建立随机森林分类
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100,criterion='entropy',max_depth=10)
data['Fare'].fillna(data.Fare.mean(),inplace=True)
data['Embarked'].fillna('S',inplace=True)
t = data.copy()
testidx = np.where(pd.isnull(t.Age)==True)[0]
trainidx = np.where(pd.isnull(t.Age)==False)[0]
testAge = t.iloc[testidx,:]
trainAge = t.iloc[trainidx,:]
knr = KNeighborsRegressor(n_neighbors=5)
knr.fit(trainAge.drop(['Age','Source','Survived','Embarked','Sex'],axis=1), trainAge.Age)
testAge.drop(['Age'],axis=1,inplace=True)
testAge['Age'] = knr.predict(testAge.drop(['Source','Survived','Embarked','Sex'],axis=1))
knn_age = pd.concat([trainAge,testAge],axis=0)

train, test = knn_age.iloc[knn_age.Source.values=='train',:].drop(['Source'],axis=1), knn_age.iloc[knn_age.Source.values=='test',:].drop(['Source','Survived'],axis=1)
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train.Embarked)
train.Embarked = le.transform(train.Embarked)
test.Embarked = le.transform(test.Embarked)

le.fit(train.Sex)
train.Sex = le.transform(train.Sex)
test.Sex = le.transform(test.Sex)

dtc = DecisionTreeClassifier(max_depth=10)
dtc.fit(train.iloc[:,:-1],train.Survived)
dtc.score(train.iloc[:,:-1],train.Survived)
test_preb = dtc.predict(test)

id = np.arange(892,892+len(test_preb),dtype=np.int).reshape(-1,1)
gender_submission = np.hstack((id,test_Survived))
np.savetxt(fname='./submission.csv', X=gender_submission,fmt='%d,%d',header='PassengerId,Survived', comments='')