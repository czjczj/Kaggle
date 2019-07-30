#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/22 17:03
#@Author: czj
#@File  : word_rnn.py

import pandas as pd
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#1. 读入数据 这里的话就是我们的一个文本
# path = "D:/MyInstallData/PyCharm/Kaggle/char_rnn/"
path = './'
raw_data = open(path+"Winston_Churchil.txt", encoding='utf-8').read()
raw_data = raw_data.lower()
#2. 将一个文本划分程为很多的句子  nltk.data.load
import nltk
sentensor = nltk.data.load("tokenizers/punkt/english.pickle")
sents = sentensor.tokenize(raw_data)
#3. 将每一个句子按照窗口大小进行组合成一个一个向量
corpus = []
for sen in sents:
    corpus.append(nltk.word_tokenize(sen))
# print(len(corpus))
# print(corpus[:3])
#4. word2vec
from gensim.models.word2vec import Word2Vec
size = 128
window = 5
min_count = 5
w2v_model = Word2Vec(corpus, size=size, window=window, min_count=min_count)

raw_input = [item for sublist in corpus for item in sublist]
# len(raw_input)

text_stream = []
for cor in raw_input:
    if (cor in w2v_model):
        text_stream.append(cor)
len(text_stream)

#5. 构造Lstm训练集合
seq_length = 10
x = []
y = []
for i in range(0, len(text_stream)-seq_length):
    given = text_stream[i:i+seq_length]
    predict = text_stream[i+seq_length]
    x.append(np.array([w2v_model[word] for word in given]))
    y.append(w2v_model[predict])
x = np.array(x).reshape(-1,seq_length,size)
y = np.array(y).reshape(-1,size)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=0)

import torch
from torch.utils.data import DataLoader, Dataset
class MyGenerator(Dataset):
    def __len__(self):
        return self.len

    def __init__(self, x, y):
        self.len = x.shape[0]
        self.datas = torch.FloatTensor(x)
        self.labels = torch.FloatTensor(y)

    def __getitem__(self, idx):
        return self.datas[idx], self.labels[idx]

#mini_batch_size
batch_size = 100
train_loader = DataLoader(dataset=MyGenerator(x_train,y_train),batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=MyGenerator(x_test,y_test),batch_size=batch_size,shuffle=True)


#6. 构造 LSTM 模型
input_size = size
hidden_size = 256
num_layers = 4
dropout = 0.2
n_output = size
learning_rate = 1e-3
n_epoch = 100

import torch.nn as nn
from torch.autograd import Variable
class MyLstm(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,dropout,n_output):
        super(MyLstm, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,\
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size,n_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        output = self.fc(output[:,-1, :])
        # print("output shape:", output.shape)
        return self.sigmoid(output)

lstm = MyLstm(input_size,hidden_size,num_layers,dropout,n_output)
criterion = nn.MSELoss()
optim = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

if torch.cuda.is_available():
    lstm.cuda()
    criterion.cuda()

def get_test_loss():
    total_loss = 0
    cnt = 0
    for i, (datas, labels) in enumerate(test_loader):
        datas = Variable(datas)
        labels = Variable(labels)
        # print(i, datas.shape)

        if torch.cuda.is_available():
            datas, labels = datas.cuda(), labels.cuda()
        # torch.cuda.empty_cache()
        output = lstm(datas)
        loss = criterion(output, labels)
        cnt += datas.shape[0]
        # print(type(cnt))
        # print(type(loss.cpu().numpy()))
        total_loss += (loss.cpu().detach().numpy()*cnt)
        # print(type(total_loss))
    return total_loss/float(cnt)

for e in range(n_epoch):
    for i, (datas, labels) in enumerate(train_loader):
        datas = Variable(datas)
        labels = Variable(labels)
        if torch.cuda.is_available():
            datas, labels = datas.cuda(), labels.cuda()

        optim.zero_grad()
        output = lstm(datas)
        loss = criterion(output, labels)
        loss.backward()
        optim.step()

        if (i+1)%100==0:
            test_loss = get_test_loss()
            print("Epoch:%d, batch_size:%d, tarin_mean_loss:%.4f, test_mean_loss:%.4f" % \
                  (e, i + 1, loss, test_loss))
            # print("Epoch:%d, batch_size:%d, tarin_mean_loss:%.4f"%(e,i+1,loss))
#7. 保存模型
torch.save(lstm, path+"word_rnn.pth")

a = torch.Tensor([1,2,3])
torch.cuda.device
a.cuda()

