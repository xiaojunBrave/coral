#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Coral 
@File    ：guet_health_evaluation_mainwindow.py
@IDE     ：PyCharm 
@Author  ：xiaoj
@Date    ：2022/11/17 10:53 AM 
"""
from PyQt5.QtWidgets import QWidget
from guet_widget.guet_mianwindow.guet_mainWindow import GuetMainWindow
from py_files.Creature import *
from py_files.Habitat import *
from py_files.WaterEnvironment import *
from py_files.BiologicalDebris import *
from py_files.Evaluate import *


class GuetHealthEvaluationWindow(GuetMainWindow):
    WE_window = None
    HB_window = None
    BLD_window = None
    CT_window = None
    """
    def __init__(self, parent):
        super(GuetHealthEvaluationWindow, self).__init__(parent)
    """

    def init_ui(self):
        super(GuetHealthEvaluationWindow, self).init_ui()
        self.button_init()

    # 按钮初始化方法
    def button_init(self):
        self.pushButton.clicked.connect(self.create_WE_Window)
        self.pushButton_2.clicked.connect(self.create_BLD_Window)
        self.pushButton_3.clicked.connect(self.create_HB_Window)
        self.pushButton_4.clicked.connect(self.create_CT_Window)
        self.pushButton_5.clicked.connect(self.create_Evaluate_Window)

    # 创建水环境版块健康评价
    def create_WE_Window(self):
        if (self.WE_window is None):
            self.WE_window = WaterEnvironment()
            self.WE_window.show()
            self.WE_window.we_signal.connect(self.get_data_we)
            self.pushButton.setChecked(True)
        else:
            self.WE_window.show()
        return self.WE_window

    # 创建栖息地版块健康评价
    def create_HB_Window(self):
        if(self.HB_window is None):
            self.HB_window = Habitat()
            self.HB_window.show()
            self.HB_window.hb_signal.connect(self.get_data_hb)
            self.pushButton_3.setChecked(True)
        else:
            self.HB_window.show()

        return self.HB_window

    # 创建生物残毒版块健康评价
    def create_BLD_Window(self):
        if (self.BLD_window is None):
            self.BLD_window = BiologicalDebris()
            self.BLD_window.show()
            self.BLD_window.bld_signal.connect(self.get_data_bld)
            self.pushButton_2.setChecked(True)
        else:
            self.BLD_window.show()

        return self.BLD_window

    # 创建生物版块健康评价
    def create_CT_Window(self):
        if (self.CT_window is None):
            self.CT_window = Creature()
            self.CT_window.show()
            self.CT_window.ct_signal.connect(self.get_data_ct)
            self.pushButton_4.setChecked(True)
        else:
            self.CT_window.show()
        return self.CT_window

    # 创建总健康评价
    def create_Evaluate_Window(self):
        self.Evaluate_window = Evaluation()
        if(self.WE_window is None and self.HB_window is None and self.BLD_window is None and self.CT_window is None):
            QMessageBox.critical(self, "错误", "无法进行珊瑚礁健康评价，缺少参数")
        else:
            if (self.we.b and self.hb.b and self.bld.b and self.ct.b):
                self.Evaluate_window.set(self.WE_window, self.HB_window, self.BLD_window, self.CT_window)
                self.Evaluate_window.show()
            else:
                QMessageBox.critical(self, "错误", "无法进行珊瑚礁健康评价，缺少参数")

    def get_data_we(self, we):
        self.we = we

    def get_data_hb(self, hb):
        self.hb = hb

    def get_data_bld(self, bld):
        self.bld = bld

    def get_data_ct(self, ct):
        self.ct = ct