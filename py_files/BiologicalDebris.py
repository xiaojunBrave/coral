# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'BiologicalDebris.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMessageBox

from utility.guet_judge import GuetJudge


class BiologicalDebris(QtWidgets.QMainWindow):
    bld_signal = pyqtSignal(object)

    def __init__(self, parent=None):
        super(BiologicalDebris, self).__init__(parent)
        self.setupUi(self)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(359, 358)
        MainWindow.setMinimumSize(QtCore.QSize(359, 358))
        MainWindow.setMaximumSize(QtCore.QSize(359, 359))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setMinimumSize(QtCore.QSize(0, 30))
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout.addWidget(self.lineEdit)
        self.horizontalLayout.setStretch(0, 6)
        self.horizontalLayout.setStretch(1, 4)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setMinimumSize(QtCore.QSize(0, 30))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.horizontalLayout_2.addWidget(self.lineEdit_2)
        self.horizontalLayout_2.setStretch(0, 6)
        self.horizontalLayout_2.setStretch(1, 4)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setMinimumSize(QtCore.QSize(0, 30))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.horizontalLayout_3.addWidget(self.lineEdit_3)
        self.horizontalLayout_3.setStretch(0, 6)
        self.horizontalLayout_3.setStretch(1, 4)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_4.setMinimumSize(QtCore.QSize(0, 30))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.horizontalLayout_4.addWidget(self.lineEdit_4)
        self.horizontalLayout_4.setStretch(0, 6)
        self.horizontalLayout_4.setStretch(1, 4)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_6.addWidget(self.label_5)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_5.setMinimumSize(QtCore.QSize(0, 30))
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.horizontalLayout_6.addWidget(self.lineEdit_5)
        self.horizontalLayout_6.setStretch(0, 6)
        self.horizontalLayout_6.setStretch(1, 4)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setContentsMargins(-1, 0, -1, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setMinimumSize(QtCore.QSize(0, 0))
        self.pushButton.setMaximumSize(QtCore.QSize(100, 50))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_5.addWidget(self.pushButton)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 359, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton.clicked.connect(self.InputEvent)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "生物残毒版块"))
        self.label.setText(_translate("MainWindow", "汞（Hg）:"))
        self.label_2.setText(_translate("MainWindow", "镉（Cd）:"))
        self.label_3.setText(_translate("MainWindow", "铅（Pb）:"))
        self.label_4.setText(_translate("MainWindow", "砷（As）:"))
        self.label_5.setText(_translate("MainWindow", "油类（oils）:"))
        self.pushButton.setText(_translate("MainWindow", "输入"))


    def InputEvent(self):
        result = QMessageBox.question(self, "注意", "您确定好输入的参数了吗",
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        hg = self.lineEdit.text()  # 汞
        cd = self.lineEdit_2.text()  # 镉
        pb = self.lineEdit_3.text()  # 铅
        shen = self.lineEdit_4.text()  # 砷
        oils = self.lineEdit_5.text()  # 油类
        b = True
        if result == QMessageBox.Yes:
            if ((len(hg) == 0) or (len(cd) == 0) or (len(pb) == 0) or (len(shen) == 0) or (len(oils) == 0)):
                hg = float(0)
                cd = float(0)
                pb = float(0)
                shen = float(0)
                oils = float(0)
                b = False
                QMessageBox.critical(self, "错误", "无法进行珊瑚礁健康评价，缺少生物残毒版块参数")
            else:
                # 判断是否输入的都是数字
                if (GuetJudge.is_number(hg) and GuetJudge.is_number(cd) and GuetJudge.is_number(pb)
                        and GuetJudge.is_number(shen) and GuetJudge.is_number(oils)):
                    QMessageBox.information(self, "通知", "成功输入数据")
                    hg = float(hg)
                    cd = float(cd)
                    pb = float(pb)
                    shen = float(shen)
                    oils = float(oils)
                    self.close()
                else:
                    QMessageBox.critical(self, "错误", "请输入正确类型的参数")
            bld = BLD(hg, cd, pb, shen, oils, b)
            self.bld_signal.emit(bld)
        elif result == QMessageBox.No:
            pass

    # 弹窗输入数据并判断是否完整输入参数,返回一个信号
    def set(self, hg, cd, pb, shen, oils, b=True):
        if ((len(hg)==0)or(len(cd)==0)or(len(pb)==0)or(len(shen)==0)or(len(oils)==0)):
            hg = float(0)
            cd = float(0)
            pb = float(0)
            shen = float(0)
            oils = float(0)
            b = False
        else:
            hg = float(hg)
            cd = float(cd)
            pb = float(pb)
            shen = float(shen)
            oils = float(oils)
        bld = BLD(hg, cd, pb, shen, oils, b)
        self.bld_signal.emit(bld)

    def evaluate(self):
        hg = float(self.lineEdit.text())  # 汞
        cd = float(self.lineEdit_2.text())  # 镉
        pb = float(self.lineEdit_3.text())  # 铅
        shen = float(self.lineEdit_4.text())  # 砷
        oils = float(self.lineEdit_5.text())  # 油类
        hg_score = 5 if (hg > 0.1) else (10 if (0.05 < hg < 0.1) else 15)
        cd_score = 5 if (cd > 2.0) else (10 if (0.2 < cd < 2.0) else 15)
        pb_score = 5 if (pb > 2.0) else (10 if (0.1 < pb < 2.0) else 15)
        shen_score = 5 if (shen > 5.0) else (10 if (1.0 < shen < 5.0) else 15)
        oils_score = 5 if (oils > 50) else (10 if (15 < oils < 50) else 15)
        bld_score = hg_score + cd_score + pb_score + shen_score + oils_score
        return bld_score/5

class BLD:
    def __init__(self, hg, cd, pb, shen, oils, b):
        self.hg = hg
        self.cd = cd
        self.pb = pb
        self.shen = shen
        self.oils = oils
        self.b = b

