#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/29 15:29
#@Author: czj
#@File  : vgg16.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# path = "D:/MyInstallData/PyCharm/Kaggle/Pytorch/data"
path = "./data"

batch_size = 500
trainset = torchvision.datasets.CIFAR10(root=path, train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root=path, train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# os.environ['TORCH_HOME'] = 'D:/OtherInstall/pytorch_models'
# model = torchvision.models.resnet152(pretrained=True)
# model = torchvision.models.vgg19_bn(pretrained=True)
model = torchvision.models.densenet161(pretrained=True)


# for param in model.parameters():
#     param.requires_grad = False
model.add_module("fc", nn.Linear(2048,10))

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

learing_rate = 1e-3
n_epoch = 100
model = model.to(device)
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learing_rate)

def get_test_acc():
    with torch.no_grad():
        total = 0
        corr = 0
        for i, (images, labels) in enumerate(testloader):
            images = torch.FloatTensor(images).to(device)
            labels = torch.LongTensor(labels).to(device)

            outputs = model(images)
            total += images.shape[0]
            corr += (torch.max(outputs,1)[1] == labels).sum()
    return corr.cpu().numpy()/total

for e in range(n_epoch):
    for i, (images, labels) in enumerate(trainloader):
        images = torch.FloatTensor(images).to(device)
        labels = torch.LongTensor(labels).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        # print(outputs.requires_grad)
        # print(outputs)
        loss = criterion(outputs, labels)
        # print(loss.requires_grad)
        loss.backward()
        optimizer.step()

        # predicts = torch.max(outputs,1)[1]
        if(i+1)%4 == 0:
            test_acc = get_test_acc()
            print("E:%d, B:%d, Test acc:%.4f"%(e, (i+1), test_acc))




