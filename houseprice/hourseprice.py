#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/11 19:55
#@Author: czj
#@File  : hourseprice.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.family'] = 'sans-serif'

test_path = 'D:\\MyInstallData\\PyCharm\\Kaggle\\houseprice\\test.csv'
train_path = 'D:\\MyInstallData\\PyCharm\\Kaggle\\houseprice\\train.csv'
train = pd.read_csv(train_path, sep=',', index_col=0)
test = pd.read_csv(test_path, sep=',',  index_col=0)

# salePrice  数据平滑  x-> log(x+1)  逆操作 x->exp(x)+1
# price = pd.DataFrame({'price':train.SalePrice, 'log(price+1)':np.log(train.SalePrice + 1)})
# price.hist()
y_train = np.log1p(train.pop('SalePrice'))
df = pd.concat([train, test], axis=0)
df['MSSubClass'] = df['MSSubClass'].astype(str)

mss = pd.get_dummies(df['MSSubClass'], prefix='MSSubClass')
pd.concat([df, mss], axis=1)