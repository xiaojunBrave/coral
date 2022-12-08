#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Coral 
@File    ：guet_coordinate_label.py
@IDE     ：PyCharm 
@Author  ：xiaoj
@Date    ：2022/11/17 9:55 AM 
"""
import math
from PyQt5 import QtGui
from PyQt5.QtGui import QBrush, QPen, QPixmap, QImage
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsLineItem, QGraphicsView, QVBoxLayout, QGraphicsTextItem, QWidget, \
    QGraphicsPixmapItem
from PyQt5.QtCore import Qt
from guet_widget.guet_label.guet_label import GuetLabel
from guet_handle.guet_coordinate_handle import GuetCoordinateHandle
from guet_widget.guet_graphics import GuetGraphicsView, GuetGraphicsPixmapItem
from PIL import  Image,ImageQt
from utility.guet_image import GuetImage

class GuetCoordinateLabel(GuetLabel):
    _scene = None
    _graphyView = None
    _x_line = None
    _y_line = None
    _x_text = None
    _y_text = None
    _x_indexs = None
    _y_indexs = None
    _imgsItem = None
    _degree_width = 50

    _pix_x = 0
    _pix_y = 0
    _geo_x = 0
    _geo_y = 0

    def __init__(self, parent=None):
        super(GuetCoordinateLabel, self).__init__(parent)
        self.bind_handle(GuetCoordinateHandle)
        self.setupUi()
        self._imgsItem = []

    def resizeEvent(self, event):
        self.refreshUI()

    def wheelEvent(self, event) -> None:
        self.zoom_graphyView(event.angleDelta().y())

    def setupUi(self):
        # 添加场景
        self._scene = QGraphicsScene()
        self._scene.setBackgroundBrush(QBrush(Qt.white))

        # 场景视图
        self._graphyView = GuetGraphicsView()
        self._graphyView.update_orgin_message.connect(self.updateOrgin)
        self._graphyView.setScene(self._scene)
        self._graphyView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._graphyView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        #self._graphyView.setDragMode(QGraphicsView.ScrollHandDrag)

        # 自动布局
        layout = QVBoxLayout()
        layout.addWidget(self._graphyView)
        self.setLayout(layout)

        self.refreshUI()

    def refreshUI(self):
        pass

    def updateOrgin(self,p_x, p_y):
        self._pix_x += p_x
        self._pix_y += p_y
        for item in self._imgsItem:
            pos = item.pos()
            item.setPos(pos.x() + p_x, pos.y() + p_y)

    def add_img(self, img):
        qim = ImageQt.ImageQt(img.image)
        pix = QPixmap.fromImage(qim).scaled(img.image.width * 0.99, img.image.height * 0.99)
        item = GuetGraphicsPixmapItem(pix)
        item.g_img = img
        if len(self._imgsItem) == 0:
            self._geo_x = img.left_top_x
            self._geo_y = img.left_top_y
            tagWidth = self._graphyView.width() / 10
            zoom = tagWidth / img.raster_x_size
            self._pix_x = (self._graphyView.width() - tagWidth) / 2
            self._pix_y = (self._graphyView.height() - self._graphyView.height() * zoom) / 2
            item.setPos(self._pix_x, self._pix_y)
            self._graphyView.scale(zoom, zoom)
        else:
            x = self._pix_x + (img.left_top_x - self._geo_x) / ((img.right_bottom_x - img.left_top_x) / (img.raster_x_size - 1))
            y = self._pix_y + (img.left_top_y - self._geo_y) / ((img.right_bottom_y - img.left_top_y) / (img.raster_y_size -1))
            item.setPos(x, y)
        self._imgsItem.append(item)
        self._scene.addItem(item)

    def remove_img(self, name):
        for item in self._scene.items():
            if isinstance(item, GuetGraphicsPixmapItem):
                if item.g_img.name == name:
                    self._scene.removeItem(item)
                    self._imgsItem.remove(item)
                    break
        if len(self._imgsItem) == 0:
            self._graphyView.resetTransform()

    def show_img_again(self, name):
        for item in self._imgsItem:
            if item.g_img.name == name:
                self._scene.addItem(item)
                return True
        return False

    def dont_show_img(self, name):
        for item in self._imgsItem:
            if item.g_img.name == name:
                self._scene.removeItem(item)
                break

    def zoom_graphyView(self, direction):
        zoom = 1.05 if direction > 0 else 0.95
        self._graphyView.scale(zoom, zoom)

    def recive_broadcast_connect(self, em):
        if em.broadcast_signal_identify == "showAImgToCoordinate":
            if not self.show_img_again(em.message_obj.name):
                self.handle.add_img(em.message_obj)
        elif em.broadcast_signal_identify == "notshowAImgToCoordinate":
            self.dont_show_img(em.message_obj.name)
        elif em.broadcast_signal_identify == "deleteAImgToCoordinate":
            self.remove_img(em.message_obj.name)




