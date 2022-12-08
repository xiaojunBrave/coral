#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CoralReefRecognitionProject 
@File ：guet_health_handle.py
@IDE  ：PyCharm 
@Author ：xiaoj
@Date ：2022/11/12 13:48 
"""

from guet_handle.guet_handle import GuetHandle
from py_files.HealthEvaluation import HealthEvaluation_Window
from guet_widget.guet_mianwindow.guet_mainWindow import GuetMainWindow

class GuetHealthHandle(GuetHandle):
    def called(self, action):
        super().called(action)
        newWindow = GuetMainWindow.create_defaultwindow(HealthEvaluation_Window)
        self.set_widget(newWindow)
        newWindow.show()



