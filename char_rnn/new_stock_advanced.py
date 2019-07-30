#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/22 20:50
#@Author: czj
#@File  : new_stock_advanced.py

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

#4. 使用corpus中的词向量进行建立word2vec 模型
from gensim.models.word2vec import Word2Vec
size = 128
w2v_model = Word2Vec(corpus_vec, size=size,window=5,min_count=5)

#5. 对于训练集合中的词  最终使用映射后的词向量表示该词
def word2vec(words):
    vec = np.zeros(size,dtype=np.float)
    count = 0
    for word in words:
        if word in w2v_model:
            vec += w2v_model[word]
            count += 1
    return vec/count
x_train = np.array([word2vec(words) for words in x_train_vec],dtype=np.float)
x_test = np.array([word2vec(words) for words in x_test_vec],dtype=np.float)

#6. 建立机器学习模型  (方法一)
# from sklearn.svm import SVC
# from sklearn.model_selection import cross_val_score
# clf = SVC(kernel='rbf', C=0.2)
# clf.fit(x_train,y_train)
# print("train acc:",clf.score(x_train, y_train))
# print("test acc:",clf.score(x_test, y_test))

#7. 卷积网络方法 （方法二）
#策略：对于每天的新闻，仅选取前 指定大小词  padding_size 个词组成的词向量，不足的0填充
#组成网络块
def transform_to_matrix(x, padding=256, vec_size=128):
    res = []
    for words in x:
        i = 0
        vec = []
        for word in words:
            if word in w2v_model:
                vec.append(list(w2v_model[word]))
                i += 1
            if i==256:
                break
        if i<256:
            vec.extend([[0]*vec_size]*(padding-i))
        res.append(vec)
    return res

x_train = transform_to_matrix(x_train_vec)
x_test = transform_to_matrix(x_test_vec)

#7.1  构建成为图片形式的输入向量 [batch_size, depth, width, height]
x_train = np.array(x_train)
x_test = np.array(x_test)
x_train = np.array(x_train).reshape(x_train.shape[0],1,\
                    x_train.shape[1], x_train.shape[2])
x_test = np.array(x_test).reshape(x_test.shape[0],1, \
                    x_test.shape[1], x_test.shape[2])

#7.2 构建网络训练
batch_size = 10
input_depth = 1
n_class = 2
learning_rate = 1e-3
epoch = 100
import torch.nn as nn
import torch
from torch.autograd import Variable
class Stock_CNN(nn.Module):
    def __init__(self, input_depth, n_class):
        super(Stock_CNN, self).__init__()
        self.c1 = nn.Conv2d(input_depth,256,kernel_size=3,stride=1,padding=1)
        self.relu = nn.ReLU()
        self.p1 = nn.MaxPool2d((2,2),stride=(2,2))
        self.c2 = nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1)
        self.p2 = nn.MaxPool2d((2,2),stride=(2,2))
        self.f1 = nn.Linear(64*32*128, 256)
        self.f2 = nn.Linear(256,n_class)
    def forward(self, x):
        x = self.relu(self.c1(x))
        x = self.p1(x)
        x = self.relu(self.c2(x))
        x = self.p2(x)
        x = self.relu(self.f1(x.view(-1,64*32*128)))
        return self.f2(x)
cnn = Stock_CNN(input_depth, n_class)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

if torch.cuda.is_available():
    cnn.cuda()
    criterion.cuda()

n_test = x_test.shape[0]
def get_test_acc():
    total = 0
    correct = 0
    for i in np.arange(0, n_test, batch_size):
        datas = x_test[i:np.min([i + batch_size, n_test]), :, :]
        labels = y_test[i:np.min([i + batch_size, n_test])]
        # if torch.cuda.is_available():
        datas, labels = datas.cuda(), labels.cuda()

        output = cnn(datas)
        correct += (torch.max(output,1)[1]==lables).sum()
        total += datas.size(0)
    return correct.cpu().numpy() / float(total)

n_train = x_train.shape[0]
for e in range(epoch):
    for i in np.arange(0,n_train,batch_size):
        datas = x_train[i:np.min([i+batch_size,n_train]),:,:]
        labels = y_train[i:np.min([i+batch_size,n_train])]

        datas = Variable(torch.FloatTensor(datas))
        labels = Variable(torch.LongTensor(labels))

        # if torch.cuda.is_available():
        datas, labels = datas.cuda(), labels.cuda()

        optim.zero_grad()
        output = cnn(datas)
        # print("output shape", output.shape)
        loss = criterion(output, labels)
        loss.backward()
        optim.step()

        test_acc = get_test_acc()
        print("Epoch:%d, batch:%d, train_loss:%.4f, test_acc:%.4f"%(e, i, loss, test_acc))

torch.save(cnn, path+"stock_cnn.pth")