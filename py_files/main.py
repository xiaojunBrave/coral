# coding:utf-8

import sys

from PyQt5.QtWidgets import QApplication, QFileDialog

from Ui_table import Ui_MainWindow
from PyQt5 import QtWidgets,QtGui,QtCore

from guet_widget.guet_mianwindow.guet_mainWindow import GuetMainWindow


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)


if __name__ == '__main__':
    myapp = QApplication(sys.argv)
    mainwindow = GuetMainWindow.create_defaultwindow(Ui_MainWindow)
    mainwindow.show()
    sys.exit(myapp.exec_())