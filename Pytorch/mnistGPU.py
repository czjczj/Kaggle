#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/15 16:46
#@Author: czj
#@File  : mnistGPU.py

import pandas as pd
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
# path = 'D:\\MyInstallData\\PyCharm\\Kaggle\\Pytorch\\'
train = datasets.MNIST(root='./mnist',train=True,
               transform=transforms.ToTensor(),
               download=False)
test = datasets.MNIST(root='./mnist',train=False,
               transform=transforms.ToTensor(),
               download=False)
# train = datasets.MNIST(root=path+'mnist',train=True,
#                transform=transforms.ToTensor(),
#                download=False)
# test = datasets.MNIST(root=path+'mnist',train=False,
#                transform=transforms.ToTensor(),
#                download=False)
#样本的个数
batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train,
                                            batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test,
                                            batch_size=batch_size,shuffle=True)

#定义卷积神经网络
class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.relu = nn.ReLU()
        self.c1 = nn.Conv2d(1,256,3,padding=1)
        self.p1 = nn.MaxPool2d((2,2))
        self.c2 = nn.Conv2d(256,128,3,padding=1)
        self.p2 = nn.MaxPool2d((2,2))
        self.f1 = nn.Linear(7*7*128,256)
        self.f2 = nn.Linear(256,10)
    def forward(self, x):
        x = self.relu(self.c1(x))
        x = self.p1(x)
        x = self.relu(self.c2(x))
        x = self.p2(x)
        x = x.view(-1,7*7*128)
        x = self.relu(self.f1(x))
        x = self.f2(x)
        return x

input_size = 28
n_epoch = 100
learning_rate = 1e-3
mnistCnn = cnn()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mnistCnn.parameters(), lr=learning_rate)

#部署到GPU
mnistCnn = mnistCnn.cuda()
criterion = criterion.cuda()

for e in range(n_epoch):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = mnistCnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('epoch:%d,batch:%d, loss:%.5f'%(e+1,i+1,loss.item()))

    total = 0
    correct = 0
    #在测试集合上验证
    for i,(images,labels) in enumerate(test_loader):
        images = Variable(images)
        labels = Variable(labels)

        print("images shape:", images.size())
        print("labels shape:", labels.size())

        images, labels = images.cuda(), labels.cuda()
        outputs = mnistCnn(images)
        _,predict = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predict==labels).sum()
    correct = correct.cpu().numpy()
    print('corr:%d, total:%d, acc:%.4f \n'%(correct,total,(correct/total)))

torch.save(mnistCnn,'mnistGPU.pth')
torch.save(mnistCnn.state_dict(),'mnistGPUparams.pth')
