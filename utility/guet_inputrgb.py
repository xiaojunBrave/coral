from PyQt5.QtCore import QFileInfo, QThread
from PyQt5.QtWidgets import QMessageBox, QFileDialog

from py_files.Rgb import RgbWindow
from utility.guet_judge import GuetJudge

class GuetInput(object):
    _rgb_window = None
    _current_widget = None
    _band_order = None
    _call_back = None
    _input_file_path = None
    _input_file_name = None
    @staticmethod
    def _open_file(title):
        QMessageBox.information(GuetInput._current_widget, "提示", "所选择文件的波段数需为3")
        file = QFileDialog.getOpenFileName(GuetInput._current_widget, title, 'C:\\', "Image files (*.tif *.dat)")
        if file is not None:
            GuetInput._input_file_path = QFileInfo(file[0]).filePath()
            GuetInput._input_file_name = QFileInfo(file[0]).fileName()
            if GuetInput._input_file_name != "":
                GuetInput._input_rgb()
        else:
            QFileDialog.exec_()

    @staticmethod
    def _input_rgb():
        GuetInput._band_order = None
        GuetInput._rgb_window = RgbWindow()
        GuetInput._rgb_window.show()
        GuetInput._rgb_window.pushButton.clicked.connect(GuetInput._InputEvent)

    @staticmethod
    def _InputEvent():
        result = QMessageBox.question(GuetInput._current_widget, "注意", "您确定好输入的参数了吗",
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        r = GuetInput._rgb_window.lineEdit.text()
        g = GuetInput._rgb_window.lineEdit_2.text()
        b = GuetInput._rgb_window.lineEdit_3.text()
        if result == QMessageBox.Yes:
            if ((len(r) == 0) or (len(g) == 0) or (len(b) == 0)):
                QMessageBox.critical(GuetInput._current_widget, "错误", "缺少相应参数")
            else:
                # 判断是否输入的都是数字
                if (GuetJudge.is_number(r) and GuetJudge.is_number(g) and GuetJudge.is_number(b)):
                    QMessageBox.information(GuetInput._current_widget, "通知", "成功输入参数")
                    r = int(r)
                    g = int(g)
                    b = int(b)
                    GuetInput._band_order = [r, g, b]
                    GuetInput._rgb_window.close()
                    GuetInput._call_back(True if GuetInput._band_order is not None else False, GuetInput._input_file_path, GuetInput._band_order)
                else:
                    QMessageBox.critical(GuetInput._current_widget, "错误", "请输入正确类型的RGB波段参数")
        elif result == QMessageBox.No:
            pass

    @staticmethod
    def Get_FilePathAndRgbOrder(current_widget, title, call_back):
        GuetInput._call_back = call_back
        GuetInput._current_widget = current_widget
        GuetInput._open_file(title)

