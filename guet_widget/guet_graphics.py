#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CoralReefRecognitionProject 
@File ：guet_graphics.py
@IDE  ：PyCharm 
@Author ：xiaoj
@Date ：2022/11/19 19:52 
"""
import sys

from PIL import ImageQt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QBrush, QPen, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsLineItem, QApplication, QMainWindow, \
    QVBoxLayout
from PyQt5.QtWidgets import QGraphicsView
from PyQt5 import QtGui, QtCore
from utility.guet_image import GuetImage


class GuetGraphicsPixmapItem(QGraphicsPixmapItem):
    g_img = None

class GuetGraphicsView(QGraphicsView):
    _max_scale = 10
    _min_scale = 0.001
    update_orgin_message = pyqtSignal(object, object)

    def __init__(self, parent=None):
        super(GuetGraphicsView, self).__init__(parent)
        self._press_x = 0
        self._press_y = 0
        self._scale = 1.0

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        pass

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        super(GuetGraphicsView, self).mousePressEvent(event)
        self._press_x, self._press_y = event.x(), event.y()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        super(GuetGraphicsView, self).mouseMoveEvent(event)
        x, y = event.x(), event.y()
        add_x, add_y = (x - self._press_x) * (1 / self._scale), (y - self._press_y) * (1 / self._scale)
        self.update_orgin_message.emit(add_x, add_y)
        self._press_x, self._press_y = x, y

    def scale(self, sx: float, sy: float) -> None:
        if self._max_scale >= self._scale * sx >= self._min_scale:
            super(GuetGraphicsView, self).scale(sx, sy)
            self._scale *= sx

    def resetTransform(self) -> None:
        super(GuetGraphicsView, self).resetTransform()
        self._scale = 1

class GuetCropImgCraphicsView(QGraphicsView):
    def __init__(self, parent=None, enable=False):
        super(GuetCropImgCraphicsView, self).__init__(parent)
        self.isEnable = enable
        self.crop_start = False
        self.start_x = -1
        self.start_y = -1
        self.end_x = -1
        self.end_y = -1
        self.lines = []
        self.img_item = None
        self.scale = 1
        self.set_ui()

    def set_ui(self):
        # 添加场景
        scene = QGraphicsScene()
        scene.setBackgroundBrush(QBrush(Qt.white))

        pen = QPen()
        pen.setWidth(1)
        pen.setColor(Qt.red)

        for i in range(4):
            line = QGraphicsLineItem()
            line.setZValue(1)
            line.setPen(pen)
            scene.addItem(line)
            self.lines.append(line)

        self.setScene(scene)
        self.update()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super(GuetCropImgCraphicsView, self).resizeEvent(event)
        self.resize_scene()

    def resize(self, a0: QtCore.QSize) -> None:
        super(GuetCropImgCraphicsView, self).resize(a0)
        self.resize_scene()

    def resize_scene(self):
        self.scene().setSceneRect(0, 0, self.size().width()-2, self.size().height()-2)


    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        super(GuetCropImgCraphicsView, self).mousePressEvent(event)
        # print("press")
        # print("scene size", self.scene().width(),self.scene().height())
        # print("graphicsView size", self.width(), self.height())

        if self.img_item:
            x_left_top, y_left_top, x_right_bottom, y_right_bottom = self.img_item.boundingRect().x(),\
                                                                     self.img_item.boundingRect().y(), \
                                                                     self.img_item.boundingRect().width(), \
                                                                     self.img_item.boundingRect().height()
            print("img_item", x_left_top, y_left_top, self.img_item.offset().x(), self.img_item.offset().y(), self.img_item.pos().x(), self.img_item.pos().y())

        self.crop_start = True
        self.start_x, self.start_y = event.x(), event.y()
        print("event pos", self.start_x, self.start_y)
        if self.img_item:
            print("img x,y,width,height", self.img_item.boundingRect().x(), self.img_item.boundingRect().y(), self.img_item.boundingRect().width(), self.img_item.boundingRect().height())

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        super(GuetCropImgCraphicsView, self).mouseReleaseEvent(event)
        self.crop_start = False

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        super(GuetCropImgCraphicsView, self).mouseMoveEvent(event)
        self.end_x, self.end_y = event.x(), event.y()
        if self.crop_start:
            self.draw_crop_rect()

    def draw_crop_rect(self):
        if self.isEnable and self._is_lines_in_imgItem():
            points = []
            for i in range(2):
                for j in range(2):
                    points.append((self.start_x if j == 0 else self.end_x, self.start_y if i == 0 else self.end_y))
            temp = points[2]
            points[2] = points[3]
            points[3] = temp
            for i in range(4):
                line = self.lines[i]
                next_point_index = (i + 1) if i < 3 else 0
                line.setLine(points[i][0], points[i][1], points[next_point_index][0], points[next_point_index][1])
        else:
            self.clear_line()

    def add_img(self, img):
        if self.img_item:
            self.scene().removeItem(self.img_item)
        scene_width, scene_height = self.scene().width(), self.scene().height()
        print("scene size", self.scene().width(), self.scene().height())
        print("img size", img.width, img.height)
        print("graphicsView size", self.width(), self.height())
        ration = 1
        height_isoffset = True
        width_isoffset = True
        if scene_height / scene_width < img.height / img.width:
            if img.height > scene_height:
                ration = scene_height / img.height
                height_isoffset = False
        else:
            if img.width > scene_width:
                ration = scene_width / img.width
                width_isoffset = False
        ration = ration if ration < 1 else 0.99
        self.scale = ration
        qim = ImageQt.ImageQt(img)
        pix = QPixmap.fromImage(qim).scaled(img.width * ration, img.height * ration)
        self.img_item = GuetGraphicsPixmapItem(pix)
        self.img_item.g_img = img

        self.img_item.setPos((scene_width - img.width * ration) / 2 if width_isoffset else 0,
                    (scene_height - img.height * ration) / 2 if height_isoffset else 0)
        self.scene().addItem(self.img_item)

    def _is_lines_in_imgItem(self):
        if self.img_item:
            x_left_top, y_left_top, x_right_bottom, y_right_bottom = self.img_item.pos().x(),\
                                                                     self.img_item.pos().y(), \
                                                                     self.img_item.pos().x() + self.img_item.boundingRect().width(), \
                                                                     self.img_item.pos().y() + self.img_item.boundingRect().height()
            if (x_left_top <= self.start_x <= x_right_bottom) and (y_left_top <= self.start_y <= y_right_bottom) \
                    and (x_left_top <= self.end_x <= x_right_bottom) and (y_left_top <= self.end_y <= y_right_bottom):
                return True

    def clear_line(self):
        for i in range(4):
            line = self.lines[i]
            line.setLine(-1, -1, -1, -1)

    def get_crop_img_indexs(self):
        """
        1、获取图片要裁剪的信息
        Returns: 裁剪开始的左上角坐标，裁剪长度、宽度

        """
        crop_start_x = ((self.start_x if self.start_x < self.end_x else self.end_x) - self.img_item.pos().x()) / self.img_item.boundingRect().width() * self.img_item.g_img.width
        crop_start_y = (self.start_y if self.start_y < self.end_y else self.end_y - self.img_item.pos().y()) / self.img_item.boundingRect().height() * self.img_item.g_img.height
        crop_x_len = abs(self.end_x - self.start_x) / self.img_item.boundingRect().width() * self.img_item.g_img.width
        crop_y_len = abs(self.end_y - self.start_y) / self.img_item.boundingRect().height() * self.img_item.g_img.height
        return crop_start_x, crop_start_y, crop_x_len, crop_y_len

    def is_ready_crop(self):
        """
        是否可以准备裁剪
        Returns:

        """
        return self._is_lines_in_imgItem()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("crop")
    window = QMainWindow()
    window.resize(500, 500)
    graphyView = GuetCropImgCraphicsView(window, enable=True)
    graphyView.resize(QtCore.QSize(500, 500))
    layout = QVBoxLayout()
    layout.addWidget(graphyView)
    window.setLayout(layout)
    window.show()
    img = GuetImage.get_rgbimgae_from_geographytif('C:/Users/Administrator/Desktop/测试图/2/test_data2.tif', [3, 1, 2])
    graphyView.add_img(img)
    sys.exit(app.exec_())
