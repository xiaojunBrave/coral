import sys

from PyQt5.QtGui import QBrush, QPixmap
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QVBoxLayout, QApplication, QMainWindow

from PIL import ImageQt
from guet_widget.guet_graphics import GuetGraphicsPixmapItem, GuetCropImgCraphicsView
from guet_widget.guet_label.guet_label import GuetLabel
from utility.guet_image import GuetImage
from PyQt5.QtCore import Qt


class GuetShowLabel(GuetLabel):
    def __init__(self, parent=None):
        super(GuetShowLabel, self).__init__(parent)
        self.setupUi()

    def setupUi(self):
        # 场景视图
        self._graphyView = GuetCropImgCraphicsView()
        # 自动布局
        layout = QVBoxLayout()
        layout.addWidget(self._graphyView)
        self.setLayout(layout)

    def add_img(self, img):
        self._graphyView.add_img(img)


if __name__ == '__main__':
    myapp = QApplication(sys.argv)
    mainwindow = QMainWindow()
    mainwindow.resize(900,900)
    guetshowlabel = GuetShowLabel(mainwindow)
    guetshowlabel.resize(900,900)
    layout = QVBoxLayout()
    layout.addWidget(guetshowlabel)
    mainwindow.setLayout(layout)
    img = GuetImage.get_rgbimgae_from_geographytif(r"C:\Users\xukejian\Desktop\test_data\2\test_data2.tif", [1,2,3])
    mainwindow.show()
    guetshowlabel.add_img(img)
    sys.exit(myapp.exec_())