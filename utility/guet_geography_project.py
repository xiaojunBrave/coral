#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Coral 
@File    ：guet_geography_project.py
@IDE     ：PyCharm 
@Author  ：xiaoj
@Date    ：2022/11/17 8:58 AM 
"""
from osgeo import gdal
from osgeo import osr
import numpy as np

class GuetGeographyProject(object):
    @staticmethod
    def get_srs_pair(dataset):
        """
        获得给定数据的投影参考系和地理参考系
        :param dataset: GDAL地理数据
        :return: 投影参考系和地理参考系
        """
        prosrs = osr.SpatialReference()
        prosrs.ImportFromWkt(dataset.GetProjection())
        geosrs = prosrs.CloneGeogCS()
        return prosrs, geosrs

    @staticmethod
    def geo2lonlat(dataset, x, y):
        """
        将投影坐标转为经纬度坐标（具体的投影坐标系由给定数据确定）
        :param dataset: GDAL地理数据
        :param x: 投影坐标x
        :param y: 投影坐标y
        :return: 投影坐标(x, y)对应的经纬度坐标(lon, lat)
        """
        prosrs, geosrs = GuetGeographyProject.get_srs_pair(dataset)
        ct = osr.CoordinateTransformation(prosrs, geosrs)
        coords = ct.TransformPoint(x, y)
        return coords[:2]


    @staticmethod
    def lonlat2geo(dataset, lon, lat):
        """
        将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
        :param dataset: GDAL地理数据
        :param lon: 地理坐标lon经度
        :param lat: 地理坐标lat纬度
        :return: 经纬度坐标(lon, lat)对应的投影坐标
        """
        prosrs, geosrs = GuetGeographyProject.get_srs_pair(dataset)
        ct = osr.CoordinateTransformation(geosrs, prosrs)
        coords = ct.TransformPoint(lon, lat)
        return coords[:2]

    @staticmethod
    def imagexy2geo(dataset, row, col):
        '''
        根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
        :param dataset: GDAL地理数据
        :param row: 像素的行号
        :param col: 像素的列号
        :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
        '''
        """
        GT(0) 左上像素左上角的x坐标。
        GT(1) w-e像素分辨率/像素宽度。
        GT(2) 行旋转（通常为零）。
        GT(3) 左上像素左上角的y坐标。
        GT(4) 列旋转（通常为零）。
        GT(5) n-s像素分辨率/像素高度（北上图像为负值）
        """
        trans = dataset.GetGeoTransform()
        px = trans[0] + col * trans[1] + row * trans[2]
        py = trans[3] + col * trans[4] + row * trans[5]
        return px, py

    @staticmethod
    def geo2imagexy(dataset, x, y):
        """
        根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
        :param dataset: GDAL地理数据
        :param x: 投影或地理坐标x
        :param y: 投影或地理坐标y
        :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
        """
        trans = dataset.GetGeoTransform()
        a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
        b = np.array([x - trans[0], y - trans[3]])
        return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解


if __name__ == '__main__':
    pass
    """
     dataset = gdal.Open("C:/Users/Administrator/Desktop/data4.tif", gdal.GA_ReadOnly)  # 返回一个gdal.Dataset类型的对象
    im_width = dataset.raster_x_size  # 栅格矩阵的列数（宽）
    im_height = dataset.raster_y_size
    print("栅格{} {}".format(im_width,im_height))
    img = GuetImage.get_rgbimgae_from_geographytif("C:/Users/Administrator/Desktop/data4.tif")
    print("img size {} {}".format(img.width,img.height))
    px, py = GuetGeographyProject.imagexy2geo(dataset,0,0)
    print("图像左上角投影坐标{} {}".format(px,py))
    px, py = GuetGeographyProject.imagexy2geo(dataset,im_width-1,im_width-1)
    print("图像右下角投影坐标{} {}".format(px,py))
    l,a = GuetGeographyProject.geo2lonlat(dataset ,px,py)
    print(l,a)
    """
