#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CoralReefRecognitionProject
@File ：guet_menubar.py
@IDE  ：PyCharm
@Author ：xiaoj
@Date ：2022/11/11 15:55
"""
from PyQt5 import  QtCore
from PyQt5.QtWidgets import QMenuBar
from guet_widget.guet_widget.guet_widget import GuetWidget
from guet_handle.guet_file_handle import GuetHandle


class GuetMenuBar(QMenuBar, GuetWidget):
    def __init__(self, parent=None):
        super(GuetMenuBar, self).__init__(parent)

