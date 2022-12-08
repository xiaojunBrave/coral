#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CoralReefRecognitionProject 
@File ：guet_file_handle.py
@IDE  ：PyCharm 
@Author ：xiaoj
@Date ：2022/11/11 21:01 
"""

from PyQt5.QtCore import QFileInfo, Qt
from PyQt5.QtGui import QIcon, QCursor

from guet_handle.guet_handle import GuetHandle
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QPushButton, QMenu, QAction

from utility.guet_inputrgb import GuetInput


class GuetFileHandle(GuetHandle):
    fname_dict = {}
    index = 0
    def called(self, action):
        super().called(action)
        if action.text() == "打开":
            self.open_file()
        elif action.text() == "关闭":
            self.close_file()

    def open_file(self):
        GuetInput.Get_FilePathAndRgbOrder(self, "打开文件", self.open)

    def open(self, tag, file_path, band_order):
        if tag:
            self.scrollArea = self.get_widget().parent().parent().scrollAreaWidgetContents
            self.scrollArea.add_button(file_path, band_order, self.get_widget().parent().parent().scrollArea)
        else:
            QMessageBox.critical(self, "错误", "请输入正确的RGB波段参数")


    def close_file(self):
        result = QMessageBox.question(self, "注意", "您确定要关闭所有影像吗",
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if result == QMessageBox.Yes:
            self.scrollArea.delete_allbutton()
        elif result == QMessageBox.No:
            pass

    def save_file(self):
        pass





