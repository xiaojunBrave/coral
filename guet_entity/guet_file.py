#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CoralReefRecognitionProject 
@File ：guet_file.py
@IDE  ：PyCharm 
@Author ：xiaoj
@Date ：2022/11/19 19:47 
"""

class GuetFile(object):
    pass

class GuetImgPath(GuetFile):
    def __init__(self, name, fullpath, band_order=None):
        self.name = name
        self.fullpath = fullpath
        self.band_order = band_order

