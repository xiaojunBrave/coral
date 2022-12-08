#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CoralReefRecognitionProject 
@File ：guet_image.py
@IDE  ：PyCharm 
@Author ：xiaoj
@Date ：2022/11/18 20:41 
"""
from PIL import Image
import cv2
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt

from utility.guet_geography_project import GuetGeographyProject


class GuetGeographyImg(object):
    def __init__(self):
        self.name = None
        # 左上角、右下角经纬度
        self.left_top_longitude = 0
        self.left_top_latitude = 0
        self.right_bottom_longitude = 0
        self.right_bottom_latitude = 0
        # 左上角、右下角投影坐标
        self.left_top_x = 0
        self.left_top_y = 0
        self.right_bottom_x = 0
        self.right_bottom_y = 0
        # 栅格数 = 图像宽高
        self.raster_x_size = 0
        self.raster_y_size = 0
        # Image
        self.image = None


class GuetImage(object):
    guetDataset = None
    @staticmethod
    def _readTif(imgPath, rgb_order):  # 读取红绿蓝（RGB）三波段数据
        dataset = gdal.Open(imgPath, gdal.GA_ReadOnly)  # 返回一个gdal.Dataset类型的对象
        return GuetImage._readtif_data(dataset, rgb_order)

    @staticmethod
    def _readtif_data(dataset, bandsOrder):
        """

        Args:
            dataset:
            bandsOrder: b4:r,b3:g,b2:b

        Returns:

        """
        if GuetImage.guetDataset is None and dataset is not None:
            GuetImage.guetDataset = dataset
        cols = dataset.RasterXSize  # tif图像的宽度
        rows = dataset.RasterYSize  # tif图像的高度
        data = np.empty([rows, cols, 3], dtype=float)  # 定义结果数组，将RGB三波段的矩阵存储
        for i in range(3):
            band = dataset.GetRasterBand(bandsOrder[i])  # 读取波段数值
            oneband_data = band.ReadAsArray()  # 读取波段数值读为numpy数组
            data[:, :, i] = oneband_data  # 将读取的结果存放在三维数组的一页三
        return data

    @staticmethod
    def _tig_to_jpg(bandData, lower_percent=0.5, higher_percent=99.5):
        """
        :param bandData:
        :param lower_percent:
        :param higher_percent:
        :return: An image object (PIL)
        """
        # banddata为读取的3、2、1波段数据
        band_Num = bandData.shape[2]  # 数组第三维度的大小，在这里是图像的通道数
        JPG_Array = np.zeros_like(bandData, dtype=np.uint8)
        for i in range(band_Num):
            minValue = 0
            maxValue = 255
            # 获取数组RGB_Array某个百分比分位上的值
            low_value = np.percentile(bandData[:, :, i], lower_percent)
            high_value = np.percentile(bandData[:, :, i], higher_percent)
            temp_value = (bandData[:, :, i] - low_value) * maxValue / (high_value - low_value)
            temp_value[temp_value < minValue] = minValue
            temp_value[temp_value > maxValue] = maxValue
            JPG_Array[:, :, i] = temp_value
        outputImg = Image.fromarray(np.uint8(JPG_Array))
        return outputImg

    @staticmethod
    def _dataset_to_rgbimg(dataset, rgb_order):
        data = GuetImage._readtif_data(dataset, rgb_order)
        img = GuetImage._tig_to_jpg(data)
        return img

    @staticmethod
    def get_rgbimgae_from_geographytif(path, band_order):
        data = GuetImage._readTif(path, band_order)
        img = GuetImage._tig_to_jpg(data)
        print("img", img)
        return img

    @staticmethod
    def get_geographytif(path, band_order):
        """
        获取tif图片信息
        Args:
            path: tif文件路径
            band_order: 数组[] 蓝绿红波段位置

        Returns: GuetGeographyImg

        """
        dataset = gdal.Open(path, gdal.GA_ReadOnly)  # 返回一个gdal.Dataset类型的对象
        geoImg = GuetGeographyImg()
        geoImg.image = GuetImage._dataset_to_rgbimg(dataset, band_order)
        geoImg.raster_x_size = dataset.RasterXSize  # tif图像的宽度
        geoImg.raster_y_size = dataset.RasterYSize  # tif图像的高度
        geoImg.left_top_x, geoImg.left_top_y = GuetGeographyProject.imagexy2geo(dataset, 0, 0)
        geoImg.right_bottom_x, geoImg.right_bottom_y = GuetGeographyProject.imagexy2geo(dataset,
                                                                                        dataset.RasterYSize - 1,
                                                                                        dataset.RasterXSize - 1
                                                                                        )
        geoImg.left_top_longitude, geoImg.left_top_latitude = GuetGeographyProject.geo2lonlat(dataset,
                                                                                              geoImg.left_top_x,
                                                                                              geoImg.left_top_y)
        geoImg.right_bottom_longitude, geoImg.right_bottom_latitude = GuetGeographyProject.geo2lonlat(dataset,
                                                                                                      geoImg.right_bottom_x,
                                                                                                      geoImg.right_bottom_y)
        del dataset
        return geoImg

    @staticmethod
    def crop_geogrphytif(source_path, destination_path, start_x, start_y, width, height):
        window = (start_x, start_y, width, height)
        gdal.Translate(destination_path, source_path,
                       srcWin=window)
    @staticmethod
    def convert_to_rgbArray(band_data, rows, cols, bandsOrder):
        data = np.empty([rows, cols, 3], dtype=float)  # 定义结果数组，将RGB三波段的矩阵存储
        for i in range(3):
            band = band_data[bandsOrder[i]-1]  # 读取波段数值
            data[:, :, i] = band  # 将读取的结果存放在三维数组的一页三
        return data

if __name__ == '__main__':
    data_path = "C:/Users/Administrator/Desktop/data.tif"
    plt.imshow(GuetImage.get_rgbimgae_from_geographytif(data_path))
    plt.show()
