#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/30 21:23
#@Author: czj
#@File  : opencv.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
from PIL import Image
path = "D:/MyInstallData/PyCharm/Kaggle/opencv/"
# path = "./"


a = cv2.imread(path+"lena.jpg")
ag = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

cv2.imshow("img", a)
cv2.imshow("img1", ag)
b = cv2.calcHist(a,[2],None,[256],[0,255])
eh = cv2.equalizeHist(ag)
enb = cv2.calcHist(eh,[2],None,[256],[0,255])
cv2.imshow("e", eh)


a,b,c = cv2.split(a)
cv2.imshow("a",a)
cv2.imshow("b",b)
cv2.imshow("c",c)

m = cv2.merge([a, b, c])
cv2.imshow("m",m)

cv2.equalizeHist()
cv2.imshow("a",cv2.resize(a,(500,500),cv2.INTER_CUBIC))


#滤波
cv2.imshow("blur", cv2.blur(a, (3,3)))
cv2.imshow("mblur", cv2.medianBlur(a, 5))
b = cv2.filter2D(ag,-1,kernel=np.ones([3,3],np.float32)/9)

a = cv2.imread(path+"pyrmeanshiftFilter.jpg")
cv2.imshow("bilaterial",cv2.bilateralFilter(a,0,10,15))
cv2.imshow("meanshiftFilter",cv2.pyrMeanShiftFiltering(a,10,10))
cv2.imshow("meanshiftFilter2",cv2.pyrMeanShiftFiltering(a,50,50))

a = cv2.imread(path+'1.png')
a = cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
cv2.imshow('a',a)

import numpy
a = cv2.imread(path+'shir.png')
x_data = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
x_data = x_data.astype(numpy.float32)
x_data = numpy.multiply(x_data, 1.0 / 255.0)

b = np.transpose(a,[2,0,1])[0,:,:]
cv2.imshow("b",b)