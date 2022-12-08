#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CoralReefRecognitionProject 
@File ：guet_mainWindow.py
@IDE  ：PyCharm 
@Author ：xiaoj
@Date ：2022/11/11 18:49 
"""
from abc import abstractmethod
from PyQt5.QtWidgets import QMainWindow

from guet_widget.guet_widget.guet_widget import GuetWidget


class GuetMainWindow(QMainWindow, GuetWidget):
    @abstractmethod
    def init_ui(self):
        """
        1、抽象方法，子类必须实现
        2、实现代码自定义创建UI
        Returns:

        """
        self.setupUi(self)

    @staticmethod
    def create_defaultwindow(window_class):
        """
        1、创建自定义的window
        Args:
            window_class:

        Returns:

        """
        w = window_class(None)
        w.init_ui()
        return w
