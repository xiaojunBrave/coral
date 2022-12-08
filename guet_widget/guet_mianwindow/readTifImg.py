#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CoralReefRecognitionProject 
@File ：readTifImg.py
@IDE  ：PyCharm 
@Author ：xiaoj
@Date ：2022/11/18 20:01 
"""
from PIL import Image
import cv2
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt


def readTif(imgPath, bandsOrder=[4, 3, 2]):  # 读取红绿蓝（RGB）三波段数据

    dataset = gdal.Open(imgPath, gdal.GA_ReadOnly)  # 返回一个gdal.Dataset类型的对象
    print("bands num:",dataset.RasterCount)
    cols = dataset.raster_x_size  # tif图像的宽度
    rows = dataset.raster_y_size  # tif图像的高度
    data = np.empty([rows, cols, 3], dtype=float)  # 定义结果数组，将RGB三波段的矩阵存储
    for i in range(3):
        band = dataset.GetRasterBand(bandsOrder[i])  # 读取波段数值
        oneband_data = band.ReadAsArray()  # 读取波段数值读为numpy数组
        data[:, :, i] = oneband_data  # 将读取的结果存放在三维数组的一页三
    return data

def tig_to_jpg(bandData, lower_percent=0.5, higher_percent=99.5):
    # banddata为读取的3、2、1波段数据
    band_Num = bandData.shape[2]           # 数组第三维度的大小，在这里是图像的通道数
    JPG_Array = np.zeros_like(bandData, dtype=np.uint8)
    for i in range(band_Num):
        minValue = 0
        maxValue = 255
        #获取数组RGB_Array某个百分比分位上的值
        low_value = np.percentile(bandData[:, :,i], lower_percent)
        high_value = np.percentile(bandData[:, :,i], higher_percent)
        temp_value =(bandData[:, :,i] - low_value) * maxValue / (high_value - low_value)
        temp_value[temp_value < minValue] = minValue
        temp_value[temp_value > maxValue] = maxValue
        JPG_Array[:, :, i] = temp_value
    outputImg = Image.fromarray(np.uint8(JPG_Array))
    return outputImg

if __name__ == '__main__':
    data = readTif("C:/Users/Administrator/Desktop/data4.tif")
    img = tig_to_jpg(data)
    print(img.width,img.height)
    #cv2.imwrite('C:/Users/Administrator/Desktop/data4.tif', img)
    plt.imshow(img)
    plt.show()
    print(img.width,img.height)