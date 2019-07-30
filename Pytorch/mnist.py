#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/13 16:46
#@Author: czj
#@File  : mnist.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
# class MyData(Dataset):
#     def __init__(self, num=10000, transform=None):
#         self.len = num
#         self.transform = transform
#     def __getitem__(self, index):
#         data = torch.rand(3,3,5) #自己的数据部分
#         label = torch.LongTensor([1]) #自己的标签部分
#         if self.transform:
#             data = self.transform(data) #自己对于数据的预处理
#         return data, label
#     def __len__(self):
#         return self.len
#
# md=MyData(transform=transforms.Normalize((0,0,0),(0.1,0.2,0.3)))
#
# dl = DataLoader(md, batch_size=4, shuffle=False, num_workers=4)


#
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
path = 'D:\\MyInstallData\\PyCharm\\Kaggle\\Pytorch\\'
# train = datasets.MNIST(root='./mnist',train=True,
#                transform=transforms.ToTensor(),
#                download=False)
# test = datasets.MNIST(root='./mnist',train=False,
#                transform=transforms.ToTensor(),
#                download=False)
train = datasets.MNIST(root=path+'mnist',train=True,
               transform=transforms.ToTensor(),
               download=False)
test = datasets.MNIST(root=path+'mnist',train=False,
               transform=transforms.ToTensor(),
               download=False)
#样本的个数
batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train,
                                            batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test,
                                            batch_size=batch_size,shuffle=True)


# input_size = 784
# hidden_size = 100
# output_size = 10
# n_epoch = 100
# learning_rate = 1e-3
# 定义全连接网络
class mnist(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(mnist,self).__init__()
        self.f1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.f2 = nn.Linear(hidden_size,output_size)

    def forward(self, x):
        x = self.f1(x)
        x = self.relu(x)
        x = self.f2(x)
        return x

#定义卷积神经网络
class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.relu = nn.ReLU()
        self.c1 = nn.Conv2d(1,50,3,padding=1)
        self.p1 = nn.MaxPool2d((2,2))
        self.c2 = nn.Conv2d(50,25,3,padding=1)
        self.p2 = nn.MaxPool2d((2,2))
        self.f1 = nn.Linear(7*7*25,100)
        self.f2 = nn.Linear(100,10)
    def forward(self, x):
        x = self.relu(self.c1(x))
        x = self.p1(x)
        x = self.relu(self.c2(x))
        x = self.p2(x)
        x = x.view(-1,7*7*25)
        x = self.relu(self.f1(x))
        x = self.f2(x)
        return x

class rnn(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, num_classes):
        super(rnn,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers,x.size(0),self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers,x.size(0),self.hidden_size))

        out, (h_n, c_n) = self.lstm(x,(h0, c0))
        # print(out.shape)
        out = self.fc(out[:, -1, :])
        # print(out.shape) ([100, 50])
        return out

class brnn(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, num_classes):
        super(brnn,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.brnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_size*2,num_classes)
    def forward(self, x):
        h0 = Variable(torch.zeros(num_layers*2,batch_size,hidden_size))
        out, h_n = self.brnn(x,h0)
        # num_layers, num_directions, batch, hidden_size
        # print(out.shape)
        rtn = self.fc(out[:, -1, :])
        return rtn



input_size = 28
hidden_size = 2
num_layers = 1
num_classes = 10
n_epoch = 100
learning_rate = 1e-3
n_epoch = 100
# m = mnist(input_size,hidden_size,output_size)
# mnistCnn = cnn()
r = rnn(input_size, hidden_size, num_layers, num_classes)
# br = brnn(input_size, hidden_size, num_layers, num_classes)
criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(r.parameters(), lr=learning_rate)

for e in range(n_epoch):
    for i, (images, labels) in enumerate(train_loader):
        # images = Variable(images.view(-1,28*28))
        # images = Variable(images)

        images = Variable(images.view(-1,28,input_size))
        # print(images.shape)
        labels = Variable(labels)
        print("labels shape:",labels.shape)
        print("labels shape:", labels.dtype)
        print("images shape:", images.shape)
        print("images shape:", images.dtype)


        optimizer.zero_grad()
        # outputs = m(images)
        # outputs = mnistCnn(images)
        outputs = r(images)
        print("outputs shape:", outputs.shape)
        print("outputs shape:", outputs.dtype)
        # outputs = br(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('epoch:%d,batch:%d, loss:%.5f'%(e,i,loss.item()))


# torch.save(mnistCnn,'model.pth')
# torch.save(mnistCnn.state_dict(),'params.pth')
torch.save(r,'rnn.pth')
torch.save(r.state_dict(),'rnnparams.pth')



# path = "D:\\MyInstallData\\PyCharm\\Kaggle\\Pytorch\\"
# newCnn = torch.load(path+'model.pth')
# # mnistCnn.load_state_dict(torch.load(path+'params.pth'))
#
# total = 0
# correct = 0
# #在测试集合上验证
# for i,(images,labels) in enumerate(test_loader):
#     # images = Variable(images.view(-1,28*28))
#     images = Variable(images)
#     labels = Variable(labels)
#     # print(i,images.shape,labels.numpy())
#     # outputs = m(images)
#     outputs = newCnn(images)
#     _,predict = torch.max(outputs,1)
#     total += labels.size(0)
#     correct += (predict==labels).numpy().sum()
#
# print('corr:%d, total:%d, acc:%.4f'%(correct,total,(correct*1.0/total)))


import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class A(nn.Module):
    def __init__(self):
        super(A, self).__init__()
        self.add_module("c1",nn.Conv2d(1,2,3,1))
        self.add_module("c2",nn.Conv2d(2,3,3,1))

a = A()
for m in a.children():
    print(m)

a = Variable(torch.randn(3,2))
F.conv1d()
F.conv2d()
F.avg_pool1d()
F.batch_norm

nn.Conv2d()


input1 = Variable(torch.randn(100, 128), requires_grad=True)
input2 = Variable(torch.randn(100, 128), requires_grad=True)
output = F.pairwise_distance(input1, input2, p=2)
output.backward()



import torchvision.datasets as dset
import torchvision.transforms as transforms

transforms.Compose([
     transforms.CenterCrop(10),
     transforms.ToTensor(),
])
from torchvision import transforms
from PIL import Image
crop = transforms.Scale(100)
path = "D:\\MyInstallData\\PyCharm\\Kaggle\\Pytorch\\lena.jpg"
img = Image.open(path)

plt.figure()
plt.subplot(2,2,1)
plt.imshow(img)
plt.show()
plt.subplot(2,2,2)
crop = transforms.Scale(100)
croped = crop(img)
plt.imshow(croped)
plt.show()
plt.subplot(2,2,3)
crop = transforms.Scale(50)
croped = crop(img)
plt.imshow(croped)
plt.show()
plt.subplot(2,2,4)
crop = transforms.Scale(25)
croped = crop(img)
plt.imshow(croped)
plt.show()

def foo(fn):
    # 定义一个嵌套函数
    def bar(*args):
        print("===1===", args)
        n = args[0]
        print("===2===", n * (n - 1))
        # 查看传给foo函数的fn函数
        print(fn.__name__)
        fn(n * (n - 1))
        print("*" * 15)
        return fn(n * (n - 1))
    return bar
'''
下面装饰效果相当于：foo(my_test)，
my_test将会替换（装饰）成该语句的返回值；
由于foo()函数返回bar函数，因此funB就是bar
'''
@foo
def my_test(a):
    print("==my_test函数==", a)
# 打印my_test函数，将看到实际上是bar函数
print(my_test) # <function foo.<locals>.bar at 0x00000000021FABF8>
# 下面代码看上去是调用my_test()，其实是调用bar()函数
my_test(10)
my_test(6, 5)