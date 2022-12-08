#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CoralReefRecognitionProject 
@File ：guet_handle.py
@IDE  ：PyCharm 
@Author ：xiaoj
@Date ：2022/11/11 20:56 
"""
from PyQt5.QtWidgets import QWidget


class GuetHandle(QWidget):
    _widget = None
    def __init__(self, parent=None):
        super(GuetHandle, self).__init__(parent)

    def called(self, action):
        """
        1、子类要根据自身业务重写此方法
        Args:
            action: 由Widget。trigger 传递过来的消息
        """
        pass

    def get_parent(self):
        return self.parent()

    def get_children(self, child_name=None):
        """
        1、根据儿子名称获取childHandle
        Args:
            child_name:

        Returns:GuetHandle 对象

        """
        child = None
        if child_name is None:
            if len(self.children()) > 0:
                child = self.childAt(0)
        else:
            self.findChild(child_name)
        return child

    def get_widget(self):
        return self._widget

    def set_widget(self, widget):
        self._widget = widget

    def connectSlot(self):
        pass






