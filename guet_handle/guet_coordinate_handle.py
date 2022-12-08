#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Coral 
@File    ：guet_coordinate_handle.py
@IDE     ：PyCharm 
@Author  ：xiaoj
@Date    ：2022/11/17 9:58 AM 
"""
from PyQt5.QtCore import pyqtSignal

from guet_handle.guet_handle import GuetHandle
from utility.guet_image import GuetImage, GuetGeographyImg


class GuetCoordinateHandle(GuetHandle):
    addImg = pyqtSignal(GuetGeographyImg)
    def __init__(self, parent=None):
        super(GuetCoordinateHandle, self).__init__(parent)
        self.imgFiles = {}

    def connectSlot(self):
        self.addImg.connect(self.get_widget().add_img)


    def add_img(self,imgInfo):
        img = GuetImage.get_geographytif(imgInfo.fullpath, imgInfo.band_order)
        img.name = imgInfo.name
        self.imgFiles[imgInfo.name] = img
        self.addImg.emit(img)


