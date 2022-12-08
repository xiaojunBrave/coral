#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CoralReefRecognitionProject 
@File ：guet_widget.py
@IDE  ：PyCharm 
@Author ：xiaoj
@Date ：2022/11/11 19:15 
"""
import os.path

from PyQt5.QtWidgets import QWidget, QScrollArea, QVBoxLayout
from PyQt5.QtCore import Qt, QFileInfo
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIcon, QCursor
from PyQt5.QtWidgets import QPushButton, QMenu, QAction

from guet_entity.guet_file import GuetImgPath
from guet_widget.guet_sigal import GuetSignalObject
class GuetWidget(QWidget):
    handle = None

    def __init__(self, parent=None):
        super(GuetWidget, self).__init__(parent)
        self.guet_sigals = []

    def bind_handle(self, handle_class):
        """
        1、绑定handle
        2、Widget中的操作信号传递给handle
        2、QWidget的业务处理在handle中完成
        Args:
            handle_class: 绑定的handle子类名
        """
        self.handle = handle_class()
        self.handle.set_widget(self)
        self.handle.setObjectName(self.objectName())
        self.handle.connectSlot()


    def call_handle(self, action):
        if self.handle is not None:
            self.handle.called(action)

    def connect_broadcast_connect(self, ms):
        if ms not in self.guet_sigals:
            if isinstance(self.parent(),GuetWidget):
                self.parent().connect_broadcast_connect(ms)
            ms.connect(self.recive_broadcast_connect)
            self.guet_sigals.append(ms)
            for c in self.children():
                if isinstance(c,GuetWidget):
                    c.connect_broadcast_connect(ms)

    def recive_broadcast_connect(self,em):
        pass

    def destroy(self, destroyWindow: bool = ..., destroySubWindows: bool = ...) -> None:
        handle = self.handle
        self.handle = None
        handle.set_widget(None)
        super(GuetWidget, self).destroy(destroyWindow, destroySubWindows)

class GuetScrollViewContent(GuetWidget):
    _fname_dict = {}
    _fbutton_dict = []

    def add_button(self, file_path, band_order, button_parent):
        layout = None
        for c in self.children():
            if isinstance(c, QVBoxLayout):
                layout = c
                break
        if layout is not None:
            layout.setAlignment(Qt.AlignTop)
            (fpath, fname) = os.path.split(file_path)
            if fname not in self._fname_dict.keys() or self._fname_dict.get(fname)==0:
                self._fname_dict.update({fname: 1})
                self.RightClickButton = GuetRightClickButton(fname, fname, file_path, band_order, button_parent)
                layout.addWidget(self.RightClickButton)
            else:
                self.RightClickButton = GuetRightClickButton(
                    fname.split('.')[0] + '(' + str(self._fname_dict.get(fname)) + ').' + str(fname.split('.')[-1]), fname, file_path, band_order, button_parent)
                self._fname_dict[fname] += 1
                layout.addWidget(self.RightClickButton)
            self._fbutton_dict.append(self.RightClickButton)
            self.RightClickButton.resize(93, 28)
            self.RightClickButton.show()
            print(self._fbutton_dict)
        else:
            pass

    def delete_button(self, button_name):
        for keyname in self._fname_dict.keys():
            if(button_name == keyname):
                self._fname_dict[keyname] = self._fname_dict[keyname] - 1

    def delete_allbutton(self):
        for i in range(len(self._fbutton_dict)-1,-1,-1):
            b = self._fbutton_dict[i]
            b.deleteSelf()
        # self._fbutton_dict.clear()


class GuetScrollArea(QScrollArea, GuetWidget):
    pass



"""
Guet_Button
"""
class GuetButton(QPushButton, GuetWidget):
    pass

class GuetRightClickButton(GuetButton):
    is_show = True
    img_signal = pyqtSignal(GuetSignalObject)
    def __init__(self, name, fname, fullpath, band_order, parent=None):
        super(GuetRightClickButton, self).__init__(parent)
        self.setText(name)
        print("button parent", self.parent().objectName())
        self.connect_broadcast_connect(self.img_signal)
        self.name = name
        self.fname = fname
        self.fullpath = fullpath
        self.band_order = band_order
        self.emit_add_img(name, fullpath, band_order)


    def mousePressEvent(self, event):
        if event.buttons() == Qt.RightButton:
            self.setContextMenuPolicy(Qt.CustomContextMenu)
            self.customContextMenuRequested.connect(self.create_rightmenu)  # 连接到菜单显示函数

    def emit_add_img(self, name, fullname, band_order):
        img_path = GuetImgPath(name, fullname, band_order)
        message = GuetSignalObject()
        message.broadcast_signal_identify = "addAImgToCoordinate"
        message.message_obj = img_path
        self.img_signal.emit(message)


    # 创建右键菜单函数
    def create_rightmenu(self):
        if(self.is_show):
            # 菜单对象
            self.right_menu = QMenu(self)
            self.actionA = QAction(QIcon('../resource/delete.JPG'), u'关闭', self)
            self.right_menu.addAction(self.actionA)
            self.actionA.triggered.connect(self.deleteSelf)
            self.actionB = QAction(QIcon('../resource/lay.JPG'), u'暂时不显示', self)
            self.right_menu.addAction(self.actionB)
            self.actionB.triggered.connect(self.not_show)
        else:
            self.right_menu = QMenu(self)
            self.actionA = QAction(QIcon('../resource/show.JPG'), u'显示', self)
            self.right_menu.addAction(self.actionA)
            self.actionA.triggered.connect(self.show)
            self.actionB = QAction(QIcon('../resource/delete.JPG'), u'关闭', self)
            self.right_menu.addAction(self.actionB)
            self.actionB.triggered.connect(self.deleteSelf)

        self.right_menu.popup(QCursor.pos())   #当鼠标在控件上右击时，在鼠标位置显示右键菜单

    def deleteSelf(self) -> None:
        "deleteAImgToCoordinate"
        message = GuetSignalObject()
        message.broadcast_signal_identify = "deleteAImgToCoordinate"
        message.message_obj = GuetImgPath(self.name, self.fullpath, self.band_order)
        self.parent().delete_button(self.fname)
        self.parent()._fbutton_dict.remove(self)
        self.deleteLater()
        self.img_signal.emit(message)


    def show(self) -> None:
        "showAImgToCoordinate"
        self.setStyleSheet("background-color: rgb(238,216,174)")
        message = GuetSignalObject()
        message.broadcast_signal_identify = "showAImgToCoordinate"
        message.message_obj = GuetImgPath(self.name, self.fullpath, self.band_order)
        self.img_signal.emit(message)
        self.is_show = True

    def not_show(self) -> None:
        "notshowAImgToCoordinate"
        self.setStyleSheet("background-color: rgb(225,225,225)")
        message = GuetSignalObject()
        message.broadcast_signal_identify = "notshowAImgToCoordinate"
        message.message_obj = GuetImgPath(self.name, self.fullpath, self.band_order)
        self.img_signal.emit(message)
        self.is_show = False




