#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/1 14:44
#@Author: czj
#@File  : biological.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#特征1776个 每个特征都是在[0,1]之间的连续变量  预测Activity(0,1)
# train_path = 'D:\\MyInstallData\\PyCharm\\Kaggle\\biological\\train.csv'
# test_path = 'D:\\MyInstallData\\PyCharm\\Kaggle\\biological\\test.csv'
# train_path = './train.csv'
# test_path = './test.csv'
# train = pd.read_csv(train_path,sep=',')
# test = pd.read_csv(test_path,sep=',')
# x1,y1 = train.iloc[:,1:],train.Activity

#样本比例
#1    2034
#0    1717
#尝试使用逻辑回归计算  得分0.51797  train acc:0.818715
def LogistcRegression():
    from sklearn.linear_model import LogisticRegressionCV
    lr = LogisticRegressionCV(penalty='l2', cv=10, verbose=3)
    lr.fit(x1, y1)
    lr.score(x1, y1)
    y_submission = lr.predict_proba(test)[:, 1]
    print("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    print("Saving Results.")
    tmp = np.vstack([range(1, len(y_submission) + 1), y_submission]).T
    np.savetxt(fname='./submission.csv', X=tmp, fmt='%d,%0.9f', header='MoleculeId,PredictedProbability', comments='')

def RandomForest():
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=1000,criterion='gini',max_depth=15,min_samples_split=10)
    rfc.fit(x1, y1)
    print('train acc:',rfc.score(x1, y1))
    print('train acc')
    y_submission = rfc.predict_proba(test)[:, 1]
    print("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    print("Saving Results.")
    tmp = np.vstack([range(1, len(y_submission) + 1), y_submission]).T
    np.savetxt(fname='./submission.csv', X=tmp, fmt='%d,%0.9f', header='MoleculeId,PredictedProbability', comments='')


##################################别人使用的方法
#logistic 回归的损失函数
def logloss(attempt, actual, epsilon=1.0e-15):
    """Logloss, i.e. the score of the bioresponse competition.
    """
    attempt = np.clip(attempt, epsilon, 1.0-epsilon)
    return - np.mean(actual * np.log(attempt) +
                     (1.0 - actual) * np.log(1.0 - attempt))
def Others():
    train_path = './train.csv'
    test_path = './test.csv'
    train = pd.read_csv(train_path,sep=',')
    test = pd.read_csv(test_path,sep=',')
    x1,y1 = train.iloc[:,1:],train.Activity
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    X, y, X_submission = x1.values,y1.values,test.values

    shuffle = True
    n_folds = 6
    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = StratifiedKFold(n_splits=n_folds)

    clfs = [RandomForestClassifier(n_estimators=1000, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=1000, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=1000, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=1000, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]

    print("Creating train and test sets for blending.")
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print(j, clf)
        dataset_blend_test_j = np.zeros((X_submission.shape[0], n_folds))
        i = 0;
        for train, test in skf.split(X,y):
            print("Fold", i)
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:, 1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:, 1]
            i = i+1
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

    print('\nBlending.')
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:, 1]

    print("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    print("Saving Results.")
    tmp = np.vstack([range(1, len(y_submission) + 1), y_submission]).T
    np.savetxt(fname='./submission.csv', X=tmp, fmt='%d,%0.9f',
               header='MoleculeId,PredictedProbability', comments='')


#XGBoost 分类
def xgboost():
    from xgboost.sklearn import XGBClassifier
    from sklearn.model_selection import GridSearchCV
    train_path = './train.csv'
    test_path = './test.csv'
    train = pd.read_csv(train_path, sep=',')
    test = pd.read_csv(test_path, sep=',')
    x1, y1 = train.iloc[:, 1:], train.Activity
    x1, y1, test = x1.values, y1.values, test.values

    params = {
        'n_estimators':np.arange(50,200,20),
        'max_depth':np.arange(8,15)
    }
    clf = XGBClassifier(n_estimators=100,objective='binary:logistic',n_jobs=15)
    gs = GridSearchCV(estimator=clf,param_grid=params,cv=5)
    print("gs start.....")
    gs.fit(x1,y1)
    y_submission = gs.predict_proba(test)[:,1]
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    print("Saving Results.")
    tmp = np.vstack([range(1, len(y_submission) + 1), y_submission]).T
    np.savetxt(fname='./submission.csv', X=tmp, fmt='%d,%0.9f',
               header='MoleculeId,PredictedProbability', comments='')


if __name__=='__main__':
    xgboost()

