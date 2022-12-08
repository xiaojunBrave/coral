#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CoralReefRecognitionProject 
@File ：guet_uimainwindow.py
@IDE  ：PyCharm 
@Author ：xiaoj
@Date ：2022/11/17 14:52 
"""
from pyqt5_plugins.examplebuttonplugin import QtGui

from guet_widget.guet_mianwindow.guet_mainWindow import GuetMainWindow


class GuetUiMainWindow(GuetMainWindow):
    def init_ui(self):
        super(GuetUiMainWindow, self).init_ui()
        pass
    def recive_broadcast_connect(self, em):
        pass

    def mouseMoveEvent(self, a0: QtGui.QMouseEvent) -> None:
        super(GuetUiMainWindow, self).mouseMoveEvent(a0)
