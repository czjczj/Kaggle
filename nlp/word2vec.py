#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/17 16:54
#@Author: czj
#@File  : word2vec.py

import pandas as pd
import numpy as np
import nltk.data
from gensim.models.word2vec import Word2Vec
import os
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

path = "D:/MyInstallData/PyCharm/Kaggle/nlp/"
d = pd.read_csv(os.path.join(path,"unlabeledTrainData.tsv"),sep='\t',escapechar='\\')

stopwords_path = os.path.join(path, 'stopwords.txt')
#将网页数据转化为 纯文本数据, 并且去掉标点（, .） 去掉停用词
def to_text(raw_example, remove_stop = False):
    example = BeautifulSoup(raw_example, 'html.parser').get_text()
    example = re.sub(r'[^a-zA-Z]', ' ', example)
    example = example.lower().split()#这样子出来的不会有多余的空格
    stopwords = []
    if remove_stop:
        stopwords = {}.fromkeys([line.rstrip() for line in open(stopwords_path)])
    w = [w for w in example if w not in stopwords]
    return ' '.join(w)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def print_call_counts(f):
    n = 0
    def wrapped(*args, **kwargs):
        nonlocal n
        n += 1
        if n % 1000 == 1:
            print('method {} called {} times'.format(f.__name__, n))
        return f(*args, **kwargs)
    return wrapped

@print_call_counts
def split_sentences(review):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = [to_text(s) for s in raw_sentences if s]
    return sentences

%time sentences = sum(d.review.apply(split_sentences), [])
print('{} reviews -> {} sentences'.format(len(d), len(sentences)))

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 设定词向量训练的参数
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

model_name = '{}features_{}minwords_{}context.model'.format(num_features, min_word_count, context)

print('Training model...')
model = Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

model.init_sims(replace=True)
model.save(os.path.join('..', 'models', model_name))

print(model.doesnt_match("man woman child kitchen".split()))
print(model.doesnt_match('france england germany berlin'.split()))

model.most_similar("man")
model.most_similar("queen")


model_name = '300features_40minwords_10context.model'
model = Word2Vec.load(os.path.join(path, model_name))