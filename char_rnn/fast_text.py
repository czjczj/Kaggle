#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/23 19:19
#@Author: czj
#@File  : fast_text.py

import pandas as pd
import numpy as np

# path = "D:/MyInstallData/PyCharm/Kaggle/char_rnn/"
path = "./"
data = pd.read_csv(path+"Combined_News_DJIA.csv")
#1. 读取数据并按照按照时间划分数据得到 train test 集合
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']

x_train = train.iloc[:,2:]
x_test = test.iloc[:,2:]
corpus = x_train.values.flatten().astype(str)
x_train = x_train.values.astype(str)
x_test = x_test.values.astype(str)
y_train = train.iloc[:,1].values
y_test = test.iloc[:,1].values
#2. 对于每一个行，我们需要得到其对应的词向量，  对于整个文本corpus, 我们需要
#每一个corpus中的词向量   x_train[row_num, :]   corpus[sentence_num, :]
from nltk.tokenize import word_tokenize
tmp_train = [" ".join(item) for item in x_train]
x_train_vec = [word_tokenize(sentence) for sentence in tmp_train]

tmp_test = [" ".join(item) for item in x_test]
x_test_vec = [word_tokenize(sentence) for sentence in tmp_test]
corpus_vec = [word_tokenize(sentence) for sentence in corpus]

#3. 对于得到的词向量，①:小写化  ②删除停顿词和数字、符号   ③ lemma 语法
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop = stopwords.words("english")
wordnet = WordNetLemmatizer()
def checkList(ls):
    rtn = []
    for word in ls:
        word = word.lower().replace("b'", '').replace('b"', '').replace('"', '').replace("'", '')
        if bool(re.search(r"\d",word)):
            continue
        if bool(re.match(r"[^\w]",word)):
            continue
        if word=='':
            continue
        if word in stop:
            continue
        rtn.append(wordnet.lemmatize(word))
    return rtn
x_train_vec = [checkList(ls) for ls in x_train_vec]
corpus_vec = [checkList(ls) for ls in corpus_vec]
x_test = [checkList(ls) for ls in x_test_vec]

for i in range(len(y_train)):
    label = '__label__' + str(y_train[i])
    x_train_vec[i].append(label)

x_train = [" ".join(x) for x in x_train_vec]
x_test = [" ".join(x) for x in x_test_vec]

with open(path+'train_ft.txt', 'w') as f:
    for sen in x_train:
        f.write(sen+'\n')

with open(path+'test_ft.txt', 'w') as f:
    for sen in x_test:
        f.write(sen+'\n')

with open(path+'test_label_ft.txt', 'w') as f:
    for label in y_test:
        f.write(str(label)+'\n')

import fasttext
clf = fasttext.train_supervised(path+'train_ft.txt',dim=256, ws=5, neg=5, epoch=100, min_count=10, lrUpdateRate=1000, bucket=200000)


y_scores = []
# 我们用predict来给出判断
labels = clf.predict(x_test)
a = [int(i[0][-1]) for i in labels[0]]
y_preds = np.array(a).flatten().astype(int)

# 我们来看看
print(len(y_test))
print(y_test)
print(len(y_preds))
print(y_preds)
from sklearn import metrics
# 算个AUC准确率 0.4705981182795699
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_preds, pos_label=1)
print(metrics.auc(fpr, tpr))

import os
import logging
logger = logging.getLogger('czj')
logging.basicConfig()