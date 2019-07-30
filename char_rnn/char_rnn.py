#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/21 8:55
#@Author: czj
#@File  : char_rnn.py

import pandas as pd
import numpy as np
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#1. 读入文本
# path = "D:/MyInstallData/PyCharm/Kaggle/char_rnn/"
path = './'
raw = open(os.path.join(path,"Winston_Churchil.txt"),encoding='utf-8').read()
raw = raw.lower()
# len(raw) #1136237

#2. 每个字符变成one-hot向量  ， 将字符和idx对应起来
ch = sorted(list(set(raw)))
idx2char = dict((i, c) for i, c in enumerate(ch))
char2idx = dict((c, i) for i, c in enumerate(ch))

#3. 做好训练集合 x, y   x字符idx列表  y的输出值idx
window = 100
x = []
y = []
for i in range(0, len(raw)-window):
    given = raw[i:i+window]
    predict = raw[i+window]
    x.append([char2idx[c] for c in given])
    y.append(char2idx[predict])

#4. 数据准备， x变成 lstm的输入形式【样本数，seq_length, input_len】 y变成one-hot向量
n_sample = len(x)
seq_length = window
x = np.array(x,dtype=np.float).reshape(n_sample,window,1)
#4.1 将x 归一化
x = x/float(len(ch))
x = x.astype(np.float32)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = enc.fit_transform(np.array(y,dtype=np.int64).reshape(-1,1)).toarray()
y = y.astype(np.int64)

#5. 构建lstm模型
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
class mylstm(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,n_classes):
        super(mylstm, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size,n_classes)

    def forward(self, x):
        #h0 = Variable(torch.zeros(self.num_layers, x.shape[0], self.hidden_size))
        #c0 = Variable(torch.zeros(self.num_layers, x.shape[0], self.hidden_size))
        # output, (h_n, c_n) = self.lstm(x,(h0,c0))
        output, (h_n, c_n) = self.lstm(x)
        output = self.fc(output[:,-1, :])
        return output
#5.1 构造训练集合
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=0)
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
class RandomDataSet(Dataset):
    def __init__(self, x, y):
        self.len = x.shape[0]
        self.data = torch.FloatTensor(x)
        self.label = torch.LongTensor(y)
    def __getitem__(self, idx):
        # print("data shape:",self.data.shape)
        # print("idx:",idx)
        return self.data[idx],self.label[idx]
    def __len__(self):
        return self.len

batch_size = 500
# train_sampler = SubsetRandomSampler(list(range(len(x_train))))
# test_sampler = SubsetRandomSampler(list(range(len(x_train))))
test_loader = DataLoader(dataset=RandomDataSet(x_test,y_test), shuffle=True, batch_size=batch_size)
train_loader = DataLoader(dataset=RandomDataSet(x_train,y_train), shuffle=True, batch_size=batch_size)

input_size = 1
hidden_size = 128
num_layers = 4
n_classes = len(ch)-1
n_epoch = 50000
learn_rate = 1e-3

lstm = mylstm(input_size, hidden_size, num_layers, n_classes)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(lstm.parameters(), lr=learn_rate)

lstm.cuda()
criterion.cuda()

def get_test_loss():
    correct = 0
    total = 0
    for i, (datas, labels) in enumerate(test_loader):
        datas = Variable(datas)
        labels = Variable(labels)
        # print("batch_size:", datas.size(0))
        datas, labels = datas.cuda(), labels.cuda()
        outputs = lstm(datas)
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predict == torch.max(labels, 1)[1]).sum()
    correct = correct.cpu().numpy()
    # print('corr:%d, total:%d, acc:%.4f \n' % (correct, total, (correct / total)))
    return correct / float(total)

for e in range(n_epoch):
    for i, (datas, labels) in enumerate(train_loader):
        datas = Variable(datas)
        labels = Variable(labels)
        datas, labels = datas.cuda(), labels.cuda()
        # print("datas device:", datas.get_device())
        # print("labels device:", labels.get_device())

        optim.zero_grad()
        #print("criterion device:", criterion.get_device())
        #print("lstm device:", lstm.get_device())
        output = lstm(datas)
        loss = criterion(output, torch.max(labels, 1)[1])
        loss.backward()
        optim.step()

        if (i+1)%1000 == 0:
            test_loss = get_test_loss()
            print("Epoch:%d, batch:%d, train_loss:%.4f, test_loss:%.4f"%(e, (i+1), loss, test_loss))
torch.save(lstm,path+'mylstm.pth')

#6. 加载得到的模型 mylstm.pth 进行预测
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
class mylstm(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,n_classes):
        super(mylstm, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size,n_classes)

    def forward(self, x):
        #h0 = Variable(torch.zeros(self.num_layers, x.shape[0], self.hidden_size))
        #c0 = Variable(torch.zeros(self.num_layers, x.shape[0], self.hidden_size))
        # output, (h_n, c_n) = self.lstm(x,(h0,c0))
        output, (h_n, c_n) = self.lstm(x)
        output = self.fc(output[:,-1, :])
        return output

import torch
path = "D:/MyInstallData/PyCharm/Kaggle/char_rnn/"
# mylstm = torch.load(os.path.join(path,"mylstm.pth"))
mylstm = torch.load(os.path.join(path,"mylstm.pth"),map_location='cpu')
window_size = 100

raw = open(os.path.join(path,"Winston_Churchil.txt"),encoding='utf-8').read()
raw = raw.lower()
ch = sorted(list(set(raw)))
idx2char = dict((i, c) for i, c in enumerate(ch))
char2idx = dict((c, i) for i, c in enumerate(ch))

def last_window(init):
    lastWindow = init[-window_size:]
    lastwindow2idx = [char2idx[i] for i in lastWindow]
    return lastwindow2idx

def next_n_num(init, times=50):
    for i in range(times):
        x_vec = last_window(init)
        # print("x_vec:",x_vec)
        # print(len(x_vec))
        x_tensor = torch.FloatTensor(np.array(x_vec).reshape(1, -1, 1))
        # print("x_tensor:",x_tensor.shape)
        next_word_idx = mylstm(x_tensor)
        # print(type(next_word_idx))
        # print(next_word_idx.shape)
        # print(next_word_idx)
        idx = (torch.max(next_word_idx,1)[1]).numpy()[0]
        next_char = idx2char[idx]
        init += next_char
        # print("next_char:", next_char)
    return init

init = "My father was--how shall I say what he was? To this day I can only\
surmise many things of him. He was a Scotchman born, and I know now that\
he had a slight Scotch accent. At the time of which I write, my early\
childhood, he was a frontiersman and hunter. I can see him now, with his\
hunting shirt and leggings and moccasins; his powder horn, engraved with\
wondrous scenes; his bullet pouch and tomahawk and hun"
init = init.lower()
out = next_n_num(init, times=100)
print("char_rnn:", out)
