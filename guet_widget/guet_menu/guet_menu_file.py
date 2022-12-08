#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CoralReefRecognitionProject 
@File ：guet_menu_file.py
@IDE  ：PyCharm 
@Author ：xiaoj
@Date ：2022/11/11 19:14 
"""

from guet_widget.guet_menu.guet_menu import GuetMenu
from guet_handle.guet_file_handle import GuetFileHandle


class GuetMenuFile(GuetMenu):
    def __init__(self, parent):
        super(GuetMenuFile, self).__init__(parent)
        self.bind_handle(GuetFileHandle)


