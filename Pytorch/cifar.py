#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/4 14:09
#@Author: czj
#@File  : cifar.py

import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#展示部分图片
# import matplotlib.pyplot as plt
# import numpy as np
# functions to show an image
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


##定义网络
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.c1 = nn.Conv2d(3,6,5)
        self.pool1 = nn.MaxPool2d((2,2))
        self.c2 = nn.Conv2d(6,16,5)
        self.pool2 = nn.MaxPool2d((2,2))
        self.f1 = nn.Linear(16*5*5,120)
        self.f2 = nn.Linear(120,84)
        self.f3 = nn.Linear(84,10)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = self.pool1(x)
        x = F.relu(self.c2(x))
        x = self.pool2(x)
        x = x.view(-1,16*5*5)
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        return self.f3(x)

import torch.optim  as optim
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

for epoch in range(2):
    run_loss = 0
    for i, data in enumerate(trainloader,0):
        input, label = data #input size(4,3,32,32) label size(4)

        optimizer.zero_grad()
        outputs = net(input)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        run_loss += loss.item()
        if i%2000==1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, run_loss / 2000))
            run_loss = 0.0
print('Finished Training')

# dataiter = iter(testloader)
# images, labels = dataiter.next()
# # imshow(torchvision.utils.make_grid(images))
# output = net(images)
#
# predicted = [1,0,2,4]
# ' '.join('%5s' % classes[predicted[j]] for j in range(4))

correct = 0
total = 0
#在测试集上输出准确率
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _,predict = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predict==labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))