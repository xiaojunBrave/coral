import numpy as np
import os
import random
from osgeo import gdal
import cv2
from sklearn.utils import shuffle
import sys

def readTiff(fileName, xoff=0, yoff=0, data_width=0, data_height=0):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize
    #  栅格矩阵的行数
    height = dataset.RasterYSize
    #  波段数
    bands = dataset.RasterCount
    #  获取数据
    if (data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
    #  获取仿射矩阵信息
    geotrans = dataset.GetGeoTransform()
    #  获取投影信息
    proj = dataset.GetProjection()
    return width, height, bands, data, geotrans, proj

def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
        # im_width, im_height, im_bands = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
        # im_width, im_height, im_bands = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

#  数据预处理：图像归一化
#  标签onehot编码
#  img为图像数据
#  label为标签数据
#  classNum为类别总数
#  buff为存储标签颜色的缓冲字典
def dataPreprocess(img, label, classNum, buff):
    #  归一化
    img = img / 255.0
    for i in range(buff.shape[0]):
        label[label == buff[i][0]] = i
    #  将数据厚度扩展到classNum层
    new_label = np.zeros(label.shape + (classNum,))
    #  将平面的label的每类，都单独变成一层
    for i in range(classNum):
        new_label[label == i, i] = 1
    label = new_label
    return (img, label)