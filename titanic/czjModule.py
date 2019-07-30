#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/5 11:08
#@Author: czj
#@File  : czjModule.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def data_info(data):
    if not isinstance(data,pd.core.frame.DataFrame):
        print('数据输入类型不正确')
        return
    print("显示列的缺失值(存在确实值的列)情况：")
    def show_na(x):
        if x.isnull().sum()>0:
            return x.isnull().sum()
    show_na = data.apply(show_na,axis=0)
    print(show_na.dropna())

    print("显示列的->属性<-情况：")
    print(data.dtypes())

def show_unique(data):
    col = data.columns.tolist()
    for name in col:
        tmp = data[name]
        print(name,tmp.value_counts().to_dict())