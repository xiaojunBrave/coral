import numpy as np
import os
import random
from osgeo import gdal
import cv2
from sklearn.utils import shuffle
from dataProcess import *

# 通道混洗操作，进行打乱训练和验证
def shuffle_skl(X,Y):
    X,Y = shuffle(X,Y, random_state= 8000)
    return X,Y

# 训练数据生成器
# batch_size为批大小
# train_image_path为训练图像路径
# train_label_path为训练标签路径
# classNum为类别总数
# buff为存储颜色的缓冲字典
# input_size为输入图像大小
def trainGenerator(batch_size, train_image_path, train_label_path, classNum, buff, input_size = None):
    imageList = os.listdir(train_image_path)
    labelList = os.listdir(train_label_path)
    # 进行打乱训练
    #imageList,labelList = shuffle_skl(imageList, labelList)
    width, height, bands, img, geotrans, proj = readTiff(train_image_path + "/" + imageList[0][:-4] + ".tif", xoff=0,yoff=0, data_width=0, data_height=0)
    # GDAL读数据是(BandNum,Width,Height)格式，要转换为(Width,Height,BandNum)格式
    img = img.swapaxes(1, 0)
    img = img.swapaxes(1, 2)
    # 生成数据
    while(True):
        img_generator = np.zeros((batch_size, img.shape[0], img.shape[1], img.shape[2]), np.uint8)
        label_generator = np.zeros((batch_size, img.shape[0], img.shape[1]), np.uint8)
        if(input_size != None):
            img_generator = np.zeros((batch_size, input_size[0], input_size[1], input_size[2]), np.uint8)
            label_generator = np.zeros((batch_size, input_size[0], input_size[1]), np.uint8)
        # 随机生成一个batch的起点
        rand = random.randint(0, len(imageList) - batch_size)
        for j in range(batch_size):
            width, height, bands, img, geotrans, proj = readTiff(train_image_path + "/" + imageList[rand + j][:-4] + ".tif",xoff=0, yoff=0, data_width=0, data_height=0)
            img = img.swapaxes(1, 0)
            img = img.swapaxes(1, 2)
            # 改变图像尺寸至特定尺寸
            img_generator[j] = img
            width2, heigh2t, bands2, label, geotrans2, proj2 = readTiff(
                train_image_path + "/" + labelList[rand + j][:-4] + ".tif", xoff=0, yoff=0, data_width=0, data_height=0)
            # 若为彩色，转为灰度
            if(len(label.shape) == 3):
                label = label.swapaxes(1, 0)
                label = label.swapaxes(1, 2)
                label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
            if(input_size != None):
                label = cv2.resize(label, (input_size[0], input_size[1]))
            label_generator[j] = label
        img_generator, label_generator = dataPreprocess(img_generator, label_generator, classNum, buff)
        yield (img_generator,label_generator)

# 验证数据生成器
# batch_size为批大小
# train_image_path为训练图像路径
# train_label_path为训练标签路径
# classNum为类别总数
# buff为存储颜色的缓冲字典
# input_size为输入图像大小
def valGenerator(batch_size, train_image_path, train_label_path, classNum, buff, input_size = None):
    imageList = os.listdir(train_image_path)
    labelList = os.listdir(train_label_path)
    # 进行打乱验证
    #imageList,labelList = shuffle_skl(imageList, labelList)
    width, height, bands, img, geotrans, proj = readTiff(train_image_path + "/" + imageList[0][:-4] + ".tif", xoff=0,yoff=0, data_width=0, data_height=0)
    # GDAL读数据是(BandNum,Width,Height)格式，要转换为(Width,Height,BandNum)格式
    img = img.swapaxes(1, 0)
    img = img.swapaxes(1, 2)
    # 生成数据
    while(True):
        img_generator = np.zeros((batch_size, img.shape[0], img.shape[1], img.shape[2]), np.uint8)
        label_generator = np.zeros((batch_size, img.shape[0], img.shape[1]), np.uint8)
        if(input_size != None):
            img_generator = np.zeros((batch_size, input_size[0], input_size[1], input_size[2]), np.uint8)
            label_generator = np.zeros((batch_size, input_size[0], input_size[1]), np.uint8)
        # 随机生成一个batch的起点
        rand = random.randint(0, len(imageList) - batch_size)
        for j in range(batch_size):
            width, height, bands, img, geotrans, proj = readTiff(train_image_path + "/" + imageList[rand + j][:-4] + ".tif",xoff=0, yoff=0, data_width=0, data_height=0)
            img = img.swapaxes(1, 0)
            img = img.swapaxes(1, 2)
            # 改变图像尺寸至特定尺寸
            img_generator[j] = img
            width2, heigh2t, bands2, label, geotrans2, proj2 = readTiff(
                train_image_path + "/" + labelList[rand + j][:-4] + ".tif", xoff=0, yoff=0, data_width=0, data_height=0)
            # 若为彩色，转为灰度
            if(len(label.shape) == 3):
                label = label.swapaxes(1, 0)
                label = label.swapaxes(1, 2)
                label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
            if(input_size != None):
                label = cv2.resize(label, (input_size[0], input_size[1]))
            label_generator[j] = label
        img_generator, label_generator = dataPreprocess(img_generator, label_generator, classNum, buff)
        yield (img_generator,label_generator)

# 测试数据生成器
# test_iamge_path为测试数据路径
# resize_shape为输入图像大小
def testGenerator(test_iamge_path, input_size = None):
    imageList = os.listdir(test_iamge_path)
    for i in range(len(imageList)):
        width, height, bands, img, geotrans, proj = readTiff(test_iamge_path + "/" + imageList[i][:-4] + ".tif",xoff=0, yoff=0, data_width=0, data_height=0)
        img = img.swapaxes(1, 0)
        img = img.swapaxes(1, 2)
        # 归一化
        img = img / 255.0
        if(input_size != None):
            # 改变图像尺寸至特定尺寸
            img = cv2.resize(img, (input_size[0], input_size[1]))
        img = np.reshape(img, (1, ) + img.shape)
        yield img