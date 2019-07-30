#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/17 15:37
#@Author: czj
#@File  : bagofword.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import nltk
from nltk.corpus import stopwords
path = "D:/MyInstallData/PyCharm/Kaggle/nlp/"
d = pd.read_csv(os.path.join(path,"labeledTrainData.tsv"),sep='\t',escapechar='\\')
print("影评数据个数：",d.shape[0])


stopwords_path = os.path.join(path, 'stopwords.txt')
#将网页数据转化为 纯文本数据, 并且去掉标点（, .） 去掉停用词
def to_text(raw_example):
    example = BeautifulSoup(raw_example, 'html.parser').get_text()
    example = re.sub(r'[^a-zA-Z]', ' ', example)
    example = example.lower().split()#这样子出来的不会有多余的空格
    stopwords = {}.fromkeys([line.rstrip() for line in open(stopwords_path)])
    w = [w for w in example if w not in stopwords]
    return ' '.join(w)
d['clean_review'] = d['review'].apply(to_text)

vectorizer = CountVectorizer(max_features=5000)
d_f = vectorizer.fit_transform(d['clean_review']).toarray()

forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(d_f, d.sentiment)
print(confusion_matrix(d.sentiment, forest.predict(d_f)))
print("train acc:",forest.score(d_f, d.sentiment))

######## 测试集合上进行分析
print("在测试集合上进行输出")
test = d = pd.read_csv(os.path.join(path,"testData.tsv"),sep='\t',escapechar='\\')
test['clean_review'] = test['review'].apply(to_text)
test_f = vectorizer.transform(test['clean_review']).toarray()
res = forest.predict(test_f)
submit = pd.DataFrame({'id':test.id,'sentiment':res})
submit.to_csv(index=False)