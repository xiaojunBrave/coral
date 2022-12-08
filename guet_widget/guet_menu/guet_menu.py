#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CoralReefRecognitionProject 
@File ：guet_menu.py
@IDE  ：PyCharm 
@Author ：xiaoj
@Date ：2022/11/12 13:35 
"""
from PyQt5.QtWidgets import QMenu
from guet_handle.guet_handle import GuetHandle
from guet_widget.guet_widget.guet_widget import GuetWidget


class GuetMenu(QMenu, GuetWidget):
    def __init__(self, parent=None):
        super(GuetMenu, self).__init__(parent)
        self.set_trigger()

    def bind_handle(self, handle_class: GuetHandle):
        super(GuetMenu, self).bind_handle(handle_class)
        self.triggered.connect(self.call_handle)

    def set_trigger(self):
        self.triggered.connect(self.action_click)

    def action_click(self, action):
        pass
