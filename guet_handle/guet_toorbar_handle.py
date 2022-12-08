import os
import shutil

import cv2
import numpy as np
from PyQt5 import QtCore
from PIL.ImageQt import ImageQt
from PyQt5.QtCore import QFileInfo
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QSlider
from matplotlib import pyplot as plt

from guet_handle.guet_handle import GuetHandle
from py_files.BrightnessAdjustment import BrightnessAdjustmentWindow
from py_files.crop import CropWindow
from utility.guet_brightness_adjust import GuetBrightnessAdjustment
from utility.guet_deeplearning import GuetDeepLearning
from py_files.LinearStretching import LinearStretchingWindow
from utility.guet_image import GuetImage
from utility.guet_inputrgb import GuetInput
from PIL import Image


class GuetToorBarHandle(GuetHandle):
    common_LinearStretching_data = "./../data/common_LinearStretching_data/common_LinearStretching_data.tif"
    common_BrightnessAdjustment_data = "./../data/common_BrightnessAdjustment_data/common_BrightnessAdjustment_data.tif"

    def called(self, action):
        super().called(action)

        if action.text() == "裁剪":
            self.open_file(1)
        elif action.text() == "亮度调整":
            self.open_file(2)
        elif action.text() == "线性拉伸":
            self.open_file(3)


    def open_file(self, i):
        if i==1:
            GuetInput.Get_FilePathAndRgbOrder(self, "请选择要裁剪的影像", self.crop)
        elif i==2:
            GuetInput.Get_FilePathAndRgbOrder(self, "请选择要亮度调整的影像", self.open_BrightnessAdjustment_window)
        elif i==3:
            GuetInput.Get_FilePathAndRgbOrder(self, "请选择要线性拉伸的影像", self.open_LinearStretching_window)

    def crop(self, tag, file_path, band_order):
        if tag:
            self.Crop_Window = CropWindow()
            self.Crop_Window.label._graphyView.resize(QtCore.QSize(1103, 721))
            self.Crop_Window.label._graphyView.isEnable = True
            img = GuetImage.get_rgbimgae_from_geographytif(file_path, band_order)
            self.Crop_Window.show()
            self.Crop_Window.label.add_img(img)

            self.file_path = file_path
            self.band_order = band_order
            self.Crop_Window.pushButton_3.clicked.connect(self.save_cropfile)
        else:
            QMessageBox.critical(self, "错误", "请输入正确的RGB波段参数")

    def open_BrightnessAdjustment_window(self, tag, file_path, band_order):
        if tag:
            self.BrightnessAdjustment_Window = BrightnessAdjustmentWindow()
            self.BrightnessAdjustment_Window.label._graphyView.resize(QtCore.QSize(1103, 721))
            # self.BrightnessAdjustment_Window.horizontalSlider.setMinimum(-10)
            # self.BrightnessAdjustment_Window.horizontalSlider.setMaximum(10)

            self.BrightnessAdjustment_Window.horizontalSlider.setMinimum(-255)
            self.BrightnessAdjustment_Window.horizontalSlider.setMaximum(255)

            # self.BrightnessAdjustment_Window.horizontalSlider.setSingleStep(0.1)
            self.BrightnessAdjustment_Window.horizontalSlider.setTickPosition(QSlider.TicksBelow)
            self.BrightnessAdjustment_Window.horizontalSlider.setTickInterval(1)

            img = GuetImage.get_rgbimgae_from_geographytif(file_path, band_order)
            # self.lightness_im_width, self.lightness_im_height, self.lightness_im_bands, self.lightness_im_data, self.lightness_im_geotrans, self.lightness_im_proj = GuetDeepLearning.readTif(file_path)
            # rgb_array = GuetImage.convert_to_rgbArray(self.lightness_im_data, self.lightness_im_height,
            #                                           self.lightness_im_width, band_order)
            self.BrightnessAdjustment_Window.show()
            self.BrightnessAdjustment_Window.label.add_img(img)

            self.file_path = file_path
            self.band_order = band_order
            self.BrightnessAdjustment_Window.horizontalSlider.valueChanged.connect(self.brightness_adjust)
            self.BrightnessAdjustment_Window.pushButton.clicked.connect(self.save_adjustfile)
        else:
            QMessageBox.critical(self, "错误", "请输入正确的RGB波段参数")

    def brightness_adjust(self):
        self.BrightnessAdjustment(self.file_path, self.BrightnessAdjustment_Window.horizontalSlider.value(), self.band_order)


    # def BrightnessAdjustment(self, value, band_order):
    #     rgb_array = GuetImage.convert_to_rgbArray(self.lightness_im_data,self.lightness_im_height,self.lightness_im_width,band_order)
    #     self.adjust_img = GuetBrightnessAdjustment.RGBAlgorithm(rgb_array, value)

    def BrightnessAdjustment(self, filePath, value, band_order):
        width, height, bands, data, geotrans, proj = GuetDeepLearning.readTif(filePath)
        img_data = GuetBrightnessAdjustment.RGBAlgorithm(data, value)


        # cv2.imwrite(self.common_BrightnessAdjustment_data, img_data)
        # width, height, bands, data2, geotrans2, proj2 = GuetDeepLearning.readTif(self.common_BrightnessAdjustment_data)
        # 将仿射矩阵信息和地图投影信息写入RGB三波段结果图中
        GuetDeepLearning.writeTiff(img_data, geotrans, proj, self.common_BrightnessAdjustment_data)
        self.adjust_img = GuetImage.get_rgbimgae_from_geographytif(self.common_BrightnessAdjustment_data, band_order)
        self.BrightnessAdjustment_Window.label.add_img(self.adjust_img)


    def open_LinearStretching_window(self, tag, file_path, band_order):
        if tag:
            self.LinearStretching_Window = LinearStretchingWindow()
            self.LinearStretching_Window.label._graphyView.resize(QtCore.QSize(1103, 721))
            img = GuetImage.get_rgbimgae_from_geographytif(file_path, band_order)
            self.LinearStretching_Window.show()
            self.LinearStretching_Window.label.add_img(img)

            self.file_path = file_path
            self.band_order = band_order
            self.LinearStretching_Window.pushButton.clicked.connect(self.button_click1)
            self.LinearStretching_Window.pushButton_2.clicked.connect(self.button_click2)
            self.LinearStretching_Window.pushButton_3.clicked.connect(self.button_click3)
        else:
            QMessageBox.critical(self, "错误", "请输入正确的RGB波段参数")

    def button_click1(self):
        self.linear_stretch(2, self.file_path, self.band_order)

    def button_click2(self):
        self.linear_stretch(5, self.file_path, self.band_order)

    def button_click3(self):
        self.save_stretchfile()

    def linear_stretch(self, i, fileName, band_order):
        width, height, bands, data, geotrans, proj = GuetDeepLearning.readTif(fileName)
        data_stretch = GuetDeepLearning.truncated_linear_stretch(data, i)
        print(data_stretch)
        GuetDeepLearning.writeTiff(data_stretch, geotrans, proj, self.common_LinearStretching_data)
        self.stretch_img = GuetImage.get_rgbimgae_from_geographytif(self.common_LinearStretching_data, band_order)
        self.LinearStretching_Window.label.add_img(self.stretch_img)
        QMessageBox.information(self, "提示", "您选择的文件已完成{}%线性拉伸".format(i))

    def save_stretchfile(self):
        result = QFileDialog.getSaveFileName(self, '选择文件要保存的路径', './../result/LinearStretching_result/', "Image files (*.tif);; Image files (*.jpg *.png)")
        if result is not None:
            SavePath = QFileInfo(result[0]).path()
            SaveName = QFileInfo(result[0]).fileName()
            if SaveName == "":
                QMessageBox.critical(self, "错误", "文件保存失败")
            else:
                self.stretch_img.save(SavePath + '/{}'.format(SaveName))
                QMessageBox.information(self, "提示", "文件已保存")
        else:
            QFileDialog.exec_()

    def save_cropfile(self):
        if(self.Crop_Window.label._graphyView.is_ready_crop()):
            crop_start_x, crop_start_y, crop_x_len, crop_y_len = self.Crop_Window.label._graphyView.get_crop_img_indexs()
            result = QFileDialog.getSaveFileName(self, '选择文件要保存的路径', './../result/Crop_result/', "Image files (*.tif)")
            if result is not None:
                SavePath = QFileInfo(result[0]).filePath()
                SaveName = QFileInfo(result[0]).fileName()
                if SaveName == "":
                    QMessageBox.critical(self, "错误", "文件保存失败")
                else:
                    GuetImage.crop_geogrphytif(self.file_path, SavePath, crop_start_x, crop_start_y, crop_x_len, crop_y_len)
                    QMessageBox.information(self, "提示", "裁剪文件已保存")
        else:
            QMessageBox.critical(self, "错误", "请在正确范围内裁剪文件")

    def save_adjustfile(self):
        result = QFileDialog.getSaveFileName(self, '选择文件要保存的路径', './../result/BrightnessAdjustment_result/', "Image files (*.tif);; Image files (*.jpg *.png)")
        if result is not None:
            SavePath = QFileInfo(result[0]).path()
            SaveName = QFileInfo(result[0]).fileName()
            if SaveName == "":
                QMessageBox.critical(self, "错误", "文件保存失败")
            else:
                self.adjust_img.save(SavePath + '/{}'.format(SaveName))
                QMessageBox.information(self, "提示", "文件已保存")
        else:
            QFileDialog.exec_()






