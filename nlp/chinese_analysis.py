#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/17 21:30
#@Author: czj
#@File  : chinese_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import jieba
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
from sklearn.externals import joblib
# path = "D:/MyInstallData/PyCharm/Kaggle/nlp/"
path = "./"
def load_data_and_preprocessing():
    neg = pd.read_excel(path+'neg.xls',header=None,index=None)
    pos = pd.read_excel(path+'pos.xls',header=None,index=None)
    #1. 结巴分词
    def word_phrase(x):
        return list(jieba.cut(x))
    neg['words'] = neg[0].apply(word_phrase)
    pos['words'] = pos[0].apply(word_phrase)
    #2. 对于给定的数据给上标签
    y = np.concatenate((np.zeros(len(neg)),np.ones(len(pos))))
    #3. 划分测试集合训练集
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((neg['words'],pos['words'])),y,test_size=0.2)
    np.save(path+'y_train.npy', y_train)
    np.save(path+'y_test.npy', y_test)
    return x_train, x_test

def build_sentence_vector(z, ndim, imdb_w2v):
    total = 0;
    words = np.zeros((1,ndim))
    for i in z:
        try:
            words += imdb_w2v[i].reshape((1,ndim))
            total += 1
        except KeyError:
            continue
    if total != 0:
        words /= total
    return words

def get_train_vecs(x_train, x_test):
    n_dim = 300
    imdb_w2v = Word2Vec(size=n_dim, min_count=20, )
    imdb_w2v.build_vocab(x_train)

    #训练模型
    imdb_w2v.train(x_train,total_examples=imdb_w2v.corpus_count,epochs=imdb_w2v.iter)

    imdb_w2v.most_similar()
    #构建 300维度特征向量模型 该300维度可以理解为 语义方面的内容
    train_vec = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_train])

    np.save(path+'train_vec.npy', train_vec)
    print("train_vec.shape:",train_vec.shape)

    #在测试集合上训练
    #同样的训练集保存起来
    #imdb_w2v.train(x_test,total_examples=imdb_w2v.corpus_count,epochs=imdb_w2v.iter)
    imdb_w2v.save(path+"w2v_model.pkl")

    #test_vec = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_test])
    np.save(path + 'test_vec.npy', test_vec)
    print("test_vec.shape:", test_vec.shape)

#SVM模型
def get_data():
    train_vecs=np.load(path+'train_vec.npy')
    y_train=np.load(path+'y_train.npy')
    test_vecs=np.load(path+'test_vec.npy')
    y_test=np.load(path+'y_test.npy')
    return train_vecs,y_train,test_vecs,y_test

def SVC():
    train_vecs, y_train, test_vecs, y_test = get_data()
    from sklearn.svm import SVC
    clf = SVC(C=0.5, verbose=True)
    clf.fit(train_vecs, y_train)
    print("train acc:", clf.score(train_vecs, y_train))
    print("test acc:", clf.score(test_vecs, y_test))
    #保存模型 sklearn.external.joblib
    joblib.dump(clf, path+"svc_model.pkl")

def get_predict_vecs(words):
    imdb_w2v = Word2Vec.load(path + "w2v_model.pkl")
    vecs = build_sentence_vector(z,300,imdb_w2v)
    return vecs

def svm_predict(string):
    words = jieba.cut(string)
    vecs = get_predict_vecs(words)
    clf = joblib.load(path+'svc_model.pkl')
    res = clf.predict(vecs)
    if int(res[0]) == 1:
        print(string+":",'positive')
    else:
        print(string + ":", 'negative')

if __name__ == "__main__":
    x_train, x_test = load_data_and_preprocessing()
    get_train_vecs(x_train, x_test)
    SVC()

    #对于单个句子进行分析
    string = '电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    svm_predict(string)

# from gensim.models.doc2vec import Doc2Vec