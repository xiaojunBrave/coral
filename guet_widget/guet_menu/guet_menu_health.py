#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CoralReefRecognitionProject 
@File ：guet_menu_health.py
@IDE  ：PyCharm 
@Author ：xiaoj
@Date ：2022/11/12 12:32 
"""
from guet_widget.guet_menu.guet_menu import GuetMenu
from guet_handle.guet_health_handle import GuetHealthHandle


class GuetMenuHealth(GuetMenu):
    def __init__(self, parent):
        super(GuetMenuHealth, self).__init__(parent)
        self.bind_handle(GuetHealthHandle)

