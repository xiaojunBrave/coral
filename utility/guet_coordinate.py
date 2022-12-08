#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CoralReefRecognitionProject 
@File ：guet_coordinate.py
@IDE  ：PyCharm 
@Author ：xiaoj
@Date ：2022/11/17 16:06 
"""
# -*-coding:utf-8 -*-
'''
#------------------------------------
@author:By yangbocsu
@file: dot.py.py
@time: 2022.03.05
#------------------------------------
'''
import matplotlib.pyplot as plt
import numpy as np
import os

#matplotlib画图中中文显示会有问题，需要这两行设置默认字体
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 0 数据准备
X = np.arange(-10,10,0.1)
Y = np.sin(X)

xmx = 12
ymx = 2

# 1 设置x,y坐标轴的刻度显示范围
fig = plt.figure()
plt.xlim(xmin = -xmx, xmax = xmx)                   # x轴的范围[-xmx,xmx]
plt.ylim(ymin = -2, ymax = ymx)                     # y轴的范围[-1,xmx]
plt.xlabel('X')
plt.ylabel('Y')
plt.title('画正弦函数')            #图的标题

# 2 画图显示 + 保存图像
# plt.scatter(X, Y, marker = 'o', alpha=0.4, color="red", label='类别A') # 画散点图
plt.plot(X,Y)   #画直线
plt.legend()                    #label='类别A' 图中显示

path = os.getcwd()              # 获取当前的工作路径
fileName = "979424151"
filePath = path + "\\" + fileName + ".png"

plt.savefig(filePath, dpi=600)   # dpi越大，图像越清晰，当然图像所占的存储也大

plt.show()
