#!/usr/bin/env python
# -*- coding:utf-8 -*-

urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'}

import multiprocessing
import requests
import os
import sys

def do(name, url):
    a = -1
    while(a != 0):
        print("%s start:"%(name))
        a = os.system("wget %s"%(url))
        if a != 0:
            print("%s failed:"%(name))
        else:
            print("%s finished:"%(name))

if __name__=="__main__":
    pool = multiprocessing.Pool(processes=5)
    for a, b in urls.items():
        pool.apply_async(do,(a,b))
    pool.close()
    pool.join()
