#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/8 15:58
#@Author: czj
#@File  : true_titanic.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.family'] = 'sans-serif'

train_path = 'D:\\MyInstallData\\PyCharm\\Kaggle\\titanic\\train.csv'
test_path = 'D:\\MyInstallData\\PyCharm\\Kaggle\\titanic\\test.csv'
train = pd.read_csv(train_path,sep=',')
test = pd.read_csv(test_path,sep=',')

##########单变量的分析
#查看数据的缺失值情况  Age, Cabin 缺失值
train.info()   #Age, Cabin 缺失值
test.info()    #Age, Cabin, Fare缺失值
#列举连续变量的数据统计情况
train.describe()
##########多个变量的分析
# Pclass0 = train[train.Survived==0]['Pclass'].value_counts()
# Pclass1 = train[train.Survived==1]['Pclass'].value_counts()
# d = pd.DataFrame({'未获救':Pclass0, '获救':Pclass1})
# d.plot(kind='bar',stacked=True)   #客舱等级对于是否获救的人数统计

# plt.scatter(train.Age, train.Survived)# 年龄与获救是否的散点图

# Sex0 = train[train.Survived==0]['Sex'].value_counts()
# Sex1 = train[train.Survived==1]['Sex'].value_counts()
# d = pd.DataFrame({'未获救':Sex0, '获救':Sex1})
# d.plot(kind='bar',stacked=True)   #Age等级对于是否获救的人数统计   可以看到在female中获救比例大

#统计每个Pclass 中年龄的分布情况
# train.Age[train.Pclass==1].plot(kind='kde')
# train.Age[train.Pclass==2].plot(kind='kde')
# train.Age[train.Pclass==3].plot(kind='kde')

#各个登录港口对于获救的影响
# Embarked0 = train[train.Survived==0]['Embarked'].value_counts()
# Embarked1 = train[train.Survived==1]['Embarked'].value_counts()
# d = pd.DataFrame({'未获救':Embarked0, '获救':Embarked1})
# d.plot(kind='bar',stacked=True)   #Age等级对于是否获救的人数统计

#统计不同船舱下不同性别的人获救的情况
# fig = plt.figure()
# fig.set(alpha=0.5)
# plt.title('不同船舱下获救的不同性别的人获救')
#
# for Pclass in range(1,4):
#     ax = fig.add_subplot(1,4,Pclass)
#     age0 = train.Sex[(train.Survived == 0) & (train.Pclass == Pclass)].value_counts()
#     age1 = train.Sex[(train.Survived == 1) & (train.Pclass == Pclass)].value_counts()
#     df = pd.DataFrame({'获救':age1,'未获救':age0})
#     df.plot(kind='bar',stacked=True)
#     ax.set_xticklabels([u"获救", u"未获救"], rotation=0)

#对于 Age 这一列做值的填充
#选择的特征 Age Pclass SibSp Parch Fare
def set_missing(train):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_validate
    ageNull = train[train.Age.isnull()][['Pclass', 'SibSp', 'Parch', 'Fare']]
    ageNotNull = train[train.Age.notnull()][['Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
    rfr = RandomForestRegressor(n_estimators=200)
    rfr.fit(ageNotNull[['Pclass', 'SibSp', 'Parch', 'Fare']],ageNotNull['Age'])
    train.loc[(train.Age.isnull()),'Age'] = rfr.predict(ageNull)
    return train, rfr

#设置 cabin 的特征关系，因为Cabin的却是值适中，且是离散型变量
def set_cabin(train):
    train.loc[train.Cabin.notnull(),'Cabin'] = 'yes'
    train.loc[train.Cabin.isnull(),'Cabin'] = 'no'
    return train

train, rfr = set_missing(train)
train = set_cabin(train)


#因为需要建立逻辑回归模型，对于train中离散型变量值，需要one-hot处理
dummy_Pclass = pd.get_dummies(train.Pclass,prefix='Pclass')
dummy_Sex = pd.get_dummies(train.Sex,prefix='Sex')
dummy_Embarked = pd.get_dummies(train.Embarked,prefix='Embarked')
dummy_Cabin = pd.get_dummies(train.Cabin,prefix='Cabin')

col = ['Survived','Age','SibSp','Parch','Fare']
train_data = pd.concat([dummy_Pclass,dummy_Sex,dummy_Embarked,dummy_Cabin,train[col].copy()],axis=1)

#对于 Age, Fare 这一类连续变量，我们需要考虑对应的取值范围和我们
#采用模型之间的关系，我们这里使用的逻辑回归，考虑到梯度下降对于收
# 敛的影响，我们采用归一化操作
#train_data[['Fare','Age']].describe()
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
age_scale = ss.fit(train_data.Age.reshape(-1,1))
train_data['Age'] = ss.fit_transform(train_data.Age.reshape(-1,1),age_scale)
fare_scale = ss.fit(train_data.Fare.reshape(-1,1))
train_data['Fare'] = ss.fit_transform(train_data.Fare.reshape(-1,1),fare_scale)

#建立逻辑回归模型进行预测
from sklearn.linear_model import LogisticRegressionCV
lr = LogisticRegressionCV(cv=20,penalty='l2')
lr.fit(train_data.drop('Survived',axis=1), train_data['Survived'])
train_score = lr.score(train_data.drop('Survived',axis=1), train_data['Survived'])
print('train acc', train_score) #0.8080808080808081

#########################对于 test 预测
testAgeNull = test[test.Age.isnull()][['Pclass', 'SibSp', 'Parch', 'Fare']]
test.loc[test.Age.isnull(),'Age'] = rfr.predict(testAgeNull)
test = set_cabin(test)
#对于 test Fare 中有一个缺失值的进行处理
test.loc[test.Fare.isnull(),'Fare'] = np.median(test.Fare.dropna().ravel())

#test 中 Age, Fare 归一化
test['Age'] = ss.fit_transform(test.Age.ravel().reshape(-1,1), age_scale)
test['Fare'] = ss.fit_transform(test.Fare.ravel().reshape(-1,1), fare_scale)

test_dummy_Pclass = pd.get_dummies(test.Pclass,prefix='Pclass')
test_dummy_Sex = pd.get_dummies(test.Sex,prefix='Sex')
test_dummy_Embarked = pd.get_dummies(test.Embarked,prefix='Embarked')
test_dummy_Cabin = pd.get_dummies(test.Cabin,prefix='Cabin')
test_col = ['Age','SibSp','Parch','Fare']
test_data = pd.concat([test_dummy_Pclass,test_dummy_Sex,test_dummy_Embarked,test_dummy_Cabin,test[test_col].copy()],axis=1)

test_Survived_hat = lr.predict(test_data)

result = pd.DataFrame({'PassengerId':test['PassengerId'].as_matrix(), 'Survived':test_Survived_hat.astype(np.int32)})
result.to_csv("./logistic_regression_predictions.csv", index=False)
#########################
"""
    coef        name
0   0.649050    Pclass_1    正相关
1   0.288254    Pclass_2    正相关
2  -0.927935    Pclass_3    负相关
3   1.339687  Sex_female    正相关
4  -1.330319    Sex_male    负相关
5   0.005251  Embarked_C    
6   0.005414  Embarked_Q
7  -0.440512  Embarked_S
8  -0.462250    Cabin_no
9   0.471618   Cabin_yes
10 -0.545084         Age
11 -0.354931       SibSp
12 -0.118144       Parch
13  0.083687        Fare
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
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
from sklearn.linear_model import LogisticRegression
tlr = LogisticRegression()
plot_learning_curve(tlr,u"学习曲线",train_data.drop('Survived',axis=1).as_matrix(), train_data['Survived'].as_matrix())
# train_score = lr.score(train_data.drop('Survived',axis=1), train_data['Survived'])

"""
当我们的模型的准确率没有提高的时候，我们可以通过交叉验证的方式，划分训练集和验证集，训练集训练以后，我们对于验证集中分类错误的数据进行手动的观察，查看哪一些数据是分错的。
"""
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import train_test_split
# lr = LogisticRegression(penalty='l1',tol=1e-6)
# X,y = train_data.drop('Survived',axis=1).as_matrix(), train_data['Survived'].as_matrix()
# cross_val_score(lr,X,y,cv=5)
#
# x1,x2,y1,y2 = train_test_split(X,y,test_size=0.3,random_state=0)
# lr.fit(x1,y1)
# lr.score(x1,y1)
#
# y2_hat = lr.predict(x2)
# x2[y2_hat != y2]

#由 plot_learning_curve 我们知道我们的模型还没过拟合，因此我们考虑可以提取更多的特征信息。
#1. Age  根据式 Mr  或者是 Miss  Mrs 进行值填充
train_path = 'D:\\MyInstallData\\PyCharm\\Kaggle\\titanic\\train.csv'
test_path = 'D:\\MyInstallData\\PyCharm\\Kaggle\\titanic\\test.csv'
train = pd.read_csv(train_path,sep=',')
test = pd.read_csv(test_path,sep=',')
nameStr = train.Name.str
missAgeMean = train[(nameStr.contains('Miss')) & train.Age.notnull()].Age.mean()
mrsAgeMean = train[(nameStr.contains('Mrs')) & train.Age.notnull()].Age.mean()
mrAgeMean = train[(nameStr.contains('Mr')) & train.Age.notnull()].Age.mean()
notAgeMean = train[~(nameStr.contains('Mr|Miss|Mrs')) & train.Age.notnull()].Age.mean()
train.loc[(nameStr.contains('Miss')) & train.Age.isnull(),'Age'] = missAgeMean
train.loc[(nameStr.contains('Mrs')) & train.Age.isnull(),'Age'] = mrsAgeMean
train.loc[(nameStr.contains('Mr')) & train.Age.isnull(),'Age'] = mrAgeMean
train.loc[~(nameStr.contains('Mr|Miss|Mrs')) & train.Age.isnull(),'Age'] = notAgeMean


"""
可以提取的更多的特征
"""
#因为Age 我们可以分成离散型的数据值表示，所处的年龄区间段落
train['AgeType'] = pd.cut(train.Age,bins=[0,12,40,60,100],labels=[1,2,3,4])
#如果Age < 12,我们判定该人可能是儿童  增加Child 字段
def isChild(x):
    if x<12:
        return 1
    else:
        return 0
train['Child'] = train['Age'].apply(isChild)
#如果一个人的名字 包含 Mrs， 并且Parch >1,我们能够判断这个人可能是一个母亲
train.loc[(train.Name.str.contains('Mrs'))&(train.Parch>1),'Mother'] = 1
train.loc[train.Mother.isnull(),'Mother'] = 0
#考虑一个大家庭对于最后预测结果的因素影响性， 将堂兄妹/兄妹个数加起来组成 Parch family_size 字段
train['family_size'] = train['Parch'] + train['SibSp']
#因为Sex 和 Pclass 是两个特别重要的字段，我们考虑进行字段的融合
#比如一个人既是  female  并且  Pclass=1 那么其获救的可能性比较的高
train['Sex_Pclass'] = train.Sex + "_" + train.Pclass.map(str)

def set_cabin(train):
    train.loc[train.Cabin.notnull(),'Cabin'] = 'yes'
    train.loc[train.Cabin.isnull(),'Cabin'] = 'no'
    return train

train = set_cabin(train)
train.Embarked.fillna('S',inplace=True)

#将训练集分成两部分进行预测
"""
['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'AgeType', 'Child',
       'Mother', 'family_size'],
"""
def categoryDeal(train):
    categ_col = ['Pclass','Sex','Embarked','AgeType','Child','Mother','Cabin']
    con = ['Survived','Age','SibSp','Parch','Fare','family_size']
    train_data = train[con]
    for i in categ_col:
        print("dummy:", i)
        dummy_tmp = pd.get_dummies(train[i],prefix=i)
        train_data = pd.concat([train_data,dummy_tmp],axis=1)
    return train_data

train_data = categoryDeal(train)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
lr = LogisticRegression(penalty='l1',tol=1e-6)
X,y = train_data.drop('Survived',axis=1).as_matrix(), train_data['Survived'].as_matrix()
x1,x2,y1,y2 = train_test_split(X,y,test_size=0.2,random_state=0)
lr.fit(x1,y1)
lr.score(x1,y1)

plot_learning_curve(lr,'Test',X,y,cv=10)
plot_learning_curve(tlr,u"学习曲线",train_data.drop('Survived',axis=1).as_matrix(), train_data['Survived'].as_matrix())

y2_hat = lr.predict(x2)
lr.score(x2,y2)
x2[y2_hat != y2]

#使用一下集成方法
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1',tol=1e-6)
bc = BaggingClassifier(base_estimator=lr,n_estimators=2000,max_samples=0.8,
                       max_features=0.8,verbose=3)
bc.fit(x1,y1)
print("train acc:",bc.score(x1,y1))#train acc: 0.8160112359550562
print("test acc:",bc.score(x2,y2))#test acc: 0.8156424581005587