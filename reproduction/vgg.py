#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/31 12:26
#@Author: czj
#@File  : vgg.py

import pandas as pd
import numpy as np

import torch as t
import torch.nn as nn
import os
import torchvision
from collections import OrderedDict
from torchvision import datasets, transforms
import torch
from torch.autograd import Variable
import logging
import multiprocessing
logger = logging.getLogger("czj")
logger.setLevel("INFO")
BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()
chlr.setFormatter(formatter)
chlr.setLevel("INFO")
# fhlr = logging.FileHandler(path+"radio.log")
# fhlr.setFormatter(formatter)
# fhlr.setLevel("INFO")
logger.addHandler(chlr)

# logger.addHandler(fhlr)

class VGG_BN(nn.Module):
    def __init__(self):
        super(VGG_BN, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1000),
        )
    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


vgg11_bn = VGG_BN()

# os.environ['TORCH_HOME'] = 'D:/OtherInstall/pytorch_models'
# b = torchvision.models.vgg11_bn()
# b.classifier[-1] = nn.Linear(4096,10)
# b.features[0] = nn.Conv2d(1,64,3,1,1)
# model = torchvision.models.resnet152()

# path = "D:/MyInstallData/PyCharm/Kaggle/reproduction/mnist"
path = './mnist'

if t.cuda.is_available():
    device = 'cuda:3'
else:
    device = 'cpu'

batch_size = 100
n_epoch = 100
learning_rate = 1e-3

vgg11_bn = vgg11_bn.to(device)
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
optimizer = t.optim.Adam(vgg11_bn.parameters(), lr=learning_rate)

mnist_trans = transforms.Compose([transforms.ToTensor()])
trainsets = datasets.MNIST(root=path, train=True, transform=mnist_trans, target_transform=None, download=False)
testsets = datasets.MNIST(root=path, train=False, transform=mnist_trans, target_transform=None, download=False)
train_loader = torch.utils.data.DataLoader(trainsets, batch_size=batch_size,shuffle=False, num_workers=multiprocessing.cpu_count())
test_loader = torch.utils.data.DataLoader(testsets, batch_size=batch_size,shuffle=False, num_workers=multiprocessing.cpu_count())



def get_test_acc():
    total = 0
    corr = 0
    for i, (imgs, labels) in enumerate(test_loader):
        imgs = Variable(imgs).to(device)
        labels = Variable(labels).to(device)

        outputs = vgg11_bn(imgs)
        corr += (t.max(outputs,1)[1]==labels).sum()
        total += imgs.shape[0]
    return corr.cpu().numpy()/float(total)


for e in range(n_epoch):
    for i, (imgs, labels) in enumerate(train_loader):
        imgs = Variable(imgs).to(device)
        print(imgs.shape)
        labels = Variable(labels).to(device)

        optimizer.zero_grad()
        outputs = vgg11_bn(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    if (i+1)%20 == 0:
        test_acc = get_test_acc()
        logger.info("E:%d, B:%d, test_acc:%.4f",e,i+1,test_acc)