#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/4 12:16
#@Author: czj
#@File  : net.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.c1 = nn.Conv2d(1,6,5)
        self.c2 = nn.Conv2d(6,16,5)
        self.f1 = nn.Linear(16*5*5,120)
        self.f2 = nn.Linear(120,84)
        self.f3 = nn.Linear(84,10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = x.view(-1, self.num_flat_feature(x))
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = self.f3(x)
        return x

    def num_flat_feature(self,x):
        size = x.size()[1:] #除去batch_size的维度
        feature_size = 1
        for i in size:
            feature_size *= i
        return feature_size


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.f1 = nn.Linear(2,2)
        self.f2 = nn.Linear(2,1)
    def forward(self, x):
        x = self.f1(x)
        x = F.relu(x)
        x = self.f2(x)
        x = F.relu(x)
        return x


input = torch.Tensor(np.array([[1,1],[1,0],[0,1],[0,0]]))
target = torch.Tensor(np.array([0,1,1,0]).reshape(-1,1))
import torch.optim as optim
net = Net()
optimizer = optim.SGD(net.parameters(),lr=0.01)
for echo in range(10000):
    output = net(input)
    optimizer.zero_grad()
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if echo%500==0:
        print(echo,loss.item())

with torch.no_grad():
    out = net(input)
    print(out)


# input = torch.randn(1,1,32,32)
# out = net(input)
# target = torch.randn(10)
# target = target.view(1,-1)
# criterion = nn.MSELoss()
#
# loss = criterion(out,target)
# print(loss)
#
# net.zero_grad()#清除图中已经存在的参数和
# net.c1.bias.grad
# loss.backward()
# net.c1.bias.grad

# learning_rate = 0.01
# for f in net.parameters():
#     f.data.sub_(f.grad.data*learning_rate)

import torch.optim as optim
optimizer = optim.SGD(net.parameters(),lr=0.01)
optimizer.zero_grad()
net = Net()
output = net()
loss = criterion(out,target)
loss.backward()
optimizer.step()

