import os
import shutil

import keras.callbacks
from PyQt5 import QtGui

from guet_widget.guet_label.guet_show_label import GuetShowLabel
from py_files.DeepLearning import DeepLearningWindow
from py_files.InputParameter import InputParameterWindow
from py_files.InputPath import InputPathWindow
from py_files.Rgb import RgbWindow
from py_files.ShowTrainingResult import ShowTrainingResultWindow
from utility.guet_deeplearning import GuetDeepLearning

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 使用CPU运行
import datetime
import math

from utility.guet_image import GuetImage
from utility.guet_inputrgb import GuetInput
from utility.guet_judge import GuetJudge
from utility.guet_train.guet_trian import GuetTrain
# 导入模型
from utility.model import unet
from PyQt5.QtCore import QFileInfo, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QWidget, QProgressBar, QVBoxLayout
from guet_handle.guet_handle import GuetHandle
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal


class CustomCallBack(keras.callbacks.Callback):
    _regiter_handle = None
    _current_step = 0

    def __init__(self):
        super(CustomCallBack, self).__init__()
        self.guet_call_back = None

    def on_predict_begin(self, logs=None):
        self._current_step = 0

    def on_predict_end(self, logs=None):
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        self._current_step += 1
        self.guet_call_back(self._current_step)

    def register_callback(self, f):
        self.guet_call_back = f


class GuetThread(QThread):
    trigger = pyqtSignal(str, object)
    step_trigger = pyqtSignal(int, int)
    img_path = None
    band_order = None

    def __init__(self):
        super(GuetThread, self).__init__()
        self.predict_callback = CustomCallBack()
        self.predict_callback.register_callback(self.progress)
        self._predict_total_step = 0

    def run(self) -> None:
        area_perc = 0.5
        # 记录测试消耗时间
        testtime = []
        # 获取当前时间
        starttime = datetime.datetime.now()

        # 权重模型路径
        model_path = GuetAlgorithmHandle._model_path
        result_path = GuetAlgorithmHandle._result_dic
        rgb_path = result_path

        # 裁剪时的步长
        RepetitiveLength = int((1 - math.sqrt(area_perc)) * 256 / 2)

        # 读取待提取影像
        im_width, im_height, im_bands, im_data, im_geotrans, im_proj = GuetDeepLearning.readTif(self.img_path)
        if im_bands >= 3:
            im_data = im_data.swapaxes(1, 0)
            im_data = im_data.swapaxes(1, 2)

            TifArray, RowOver, ColumnOver = GuetDeepLearning.TifCroppingArray(im_data, RepetitiveLength)

            # 记录开始裁剪的时间
            endtime = datetime.datetime.now()
            text = "读取tif并裁剪预处理完毕,目前耗时间: " + str((endtime - starttime).seconds) + "s"
            print(text)
            testtime.append(text)

            # 调用模型，这里的None表示没有预训练权重，因为我们的模型里没有使用其他预训练的模型，因此这里写为None就行
            model = unet(None)
            # .load_weights为调用已经训练好的权重
            model.load_weights(model_path)

            self._predict_total_step = len(TifArray) * len(TifArray[0])

            testGene = GuetDeepLearning.testGenerator(TifArray)
            results = model.predict_generator(testGene,
                                              self._predict_total_step,
                                              verbose=1,
                                              callbacks=self.predict_callback)
            # 获取结果矩阵
            result_shape = (im_data.shape[0], im_data.shape[1])
            result_data = GuetDeepLearning.Result(result_shape, TifArray, results, RepetitiveLength, RowOver,
                                                  ColumnOver)

            # 保存结果，这里的结果为单波段影像
            GuetDeepLearning.writeTiff(result_data, im_geotrans, im_proj, result_path)
            # 改为RGB三波段
            GuetDeepLearning.ThreeBand(result_path, rgb_path)
            GuetDeepLearning.RGB(rgb_path)

            # 读取RGB三波结果图
            width3, height3, bands3, data3, geotrans3, proj3 = GuetDeepLearning.readTif(rgb_path, xoff=0, yoff=0, data_width=0,
                                                                       data_height=0)
            # 将仿射矩阵信息和地图投影信息写入RGB三波段结果图中
            GuetDeepLearning.writeTiff(data3, im_geotrans, im_proj, rgb_path)
            self.trigger.emit(rgb_path, self.band_order)

    def progress(self, current_step):
        self.step_trigger.emit(self._predict_total_step, current_step)


class GuetAlgorithmHandle(GuetHandle):
    _model_path = "./../utility/weight/unet_model.hdf5"
    _result_dic = "./../data/common_DeepLearning_data/common_DeepLearning_data.tif"

    # 预测分割图片的数量
    _predict_total_step = 0

    def __init__(self, parent=None):
        super(GuetAlgorithmHandle, self).__init__(parent)
        self.predict_total_step = 0
        self.trainingmodel_summary = None
        self.inputpath_isok = True
        self.guettrain = None

    def called(self, action):
        super().called(action)
        if action.text() == "深度学习方法训练":
            self.open_inputPath_window()
        elif action.text() == "深度学习方法提取":
            self.open_file()

    def open_inputPath_window(self):
        self.iuputpath_window = InputPathWindow()
        self.iuputpath_window.show()
        self.iuputpath_window.pushButton_3.clicked.connect(self.button_click1)
        self.iuputpath_window.pushButton_4.clicked.connect(self.button_click2)
        self.iuputpath_window.pushButton_5.clicked.connect(self.button_click3)
        self.iuputpath_window.pushButton_6.clicked.connect(self.button_click4)
        self.iuputpath_window.pushButton_7.clicked.connect(self.button_click5)
        self.iuputpath_window.pushButton.clicked.connect(self.input_parameter)
        self.iuputpath_window.pushButton_2.clicked.connect(self.close_iuputPath_window)

    def button_click1(self):
        self.inputpath(1)

    def button_click2(self):
        self.inputpath(2)

    def button_click3(self):
        self.inputpath(3)

    def button_click4(self):
        self.inputpath(4)

    def button_click5(self):
        self.inputpath(5)

    def inputpath(self, i):
        if i == 1:
            save_result = QFileDialog.getSaveFileName(self, '选择权重文件要保存的路径', './../utility/weight/', 'Image files (*.hdf5 *.h5)')
            if save_result is not None:
                SavePath = QFileInfo(save_result[0]).filePath()
                SaveName = QFileInfo(save_result[0]).fileName()
                if SaveName == "":
                    QMessageBox.critical(self, "错误", "路径输入失败")
                    self.inputpath_isok = False
                else:
                    self.model_path = SavePath
                    QMessageBox.information(self, "提示", "路径输入成功")
                    self.iuputpath_window.pushButton_3.setText(".../" + SavePath.split('/')[-1])
                    self.inputpath_isok = True
            else:
                QFileDialog.exec_()
        else:
            result = QFileDialog.getExistingDirectory(self, '选择文件夹', 'C:\\')
            if result is not None:
                if i == 2:
                    self.train_image_path = result
                    self.iuputpath_window.pushButton_4.setText(".../" + self.train_image_path.split('/')[-1])
                elif i == 3:
                    self.train_label_path = result
                    self.iuputpath_window.pushButton_5.setText(".../" + self.train_label_path.split('/')[-1])
                elif i == 4:
                    self.verify_image_path = result
                    self.iuputpath_window.pushButton_6.setText(".../" + self.verify_image_path.split('/')[-1])
                elif i == 5:
                    self.verify_label_path = result
                    self.iuputpath_window.pushButton_7.setText(".../" + self.verify_label_path.split('/')[-1])
                QMessageBox.information(self, "提示", "路径输入成功")
                self.inputpath_isok = True
            else:
                QMessageBox.critical(self, "错误", "路径输入失败")
                self.inputpath_isok = True

    def input_parameter(self):
        self.iuputpath_window.close()
        if self.inputpath_isok:
            self.iuputparameter_window = InputParameterWindow()
            self.iuputparameter_window.show()
            self.iuputparameter_window.pushButton.clicked.connect(self.training)
            self.iuputparameter_window.pushButton_2.clicked.connect(self.close_iuputParameter_window)
            self.iuputparameter_window.pushButton_3.clicked.connect(self.input_pretraining_path)
        else:
            QMessageBox.critical(self, "错误", "路径输入失败")

    def training(self):
        self.batch_size = self.iuputparameter_window.lineEdit.text()
        self.categories_number = self.iuputparameter_window.lineEdit_2.text()
        self.img_width = self.iuputparameter_window.lineEdit_3.text()
        self.img_height = self.iuputparameter_window.lineEdit_5.text()
        self.band_number = self.iuputparameter_window.lineEdit_4.text()
        self.training_rounds = self.iuputparameter_window.lineEdit_6.text()
        self.initial_learningrate = self.iuputparameter_window.lineEdit_7.text()
        self.pretraining_path = None
        self.iuputparameter_window.close()

        self.guettrain = GuetTrain()
        self.guettrain.model_summary.connect(self.get_model_summary)
        self.show_trainingresult_window = ShowTrainingResultWindow()
        self.show_trainingresult_window.show()
        self.show_trainingresult_window.textBrowser.setStyleSheet("background-color:black;color:white")
        self.guettrain.train(self.model_path, self.train_image_path, self.train_label_path,
                         self.verify_image_path,self.verify_label_path, int(self.batch_size),
                         int(self.categories_number), (int(self.img_width),int(self.img_height),int(self.band_number)),
                         int(self.training_rounds), float(self.initial_learningrate), self.pretraining_path)
        # self.guettrain.train('','','','','',0,0,(0,0,0),0,0,'')

    def input_pretraining_path(self):
        result = QFileDialog.getSaveFileName(self, '选择预训练权重路径', './../utility/weight/',
                                                  'Image files (*.hdf5 *.h5)')
        if result is not None:
            self.pretraining_path = QFileInfo(result[0]).filePath()
            self.pretraining_Name = QFileInfo(result[0]).fileName()
            if self.SaveName == "":
                QMessageBox.critical(self, "错误", "路径输入失败")
            else:
                QMessageBox.information(self, "提示", "路径输入成功")
        else:
            QFileDialog.exec_()

    def close_iuputPath_window(self):
        self.iuputpath_window.close()

    def close_iuputParameter_window(self):
        self.iuputparameter_window.close()

    def open_file(self):
        GuetInput.Get_FilePathAndRgbOrder(self, "请选择待提取的影像", self.open_deepLearning_window)

    def open_deepLearning_window(self, tag, file_path, band_order):
        if tag:
            self.DeepLearning_Window = DeepLearningWindow()
            self.DeepLearning_Window.show()
            self.DeepLearning_Window.pgb = QProgressBar(self.DeepLearning_Window)
            self.DeepLearning_Window.pgb.setGeometry(QtCore.QRect(400, 280, 351, 91))
            self.DeepLearning_Window.pgb.setStyleSheet("QProgressBar {text-align:center; font-size:20px}")
            self.DeepLearning_Window.pgb.show()
            self.deepLearning_algorithm(file_path, band_order)
        else:
            QMessageBox.critical(self, "错误", "请输入正确的RGB波段参数")

    def deepLearning_algorithm(self, img_path, band_order):
        t = GuetThread()
        t.img_path = img_path
        t.band_order = band_order
        t.trigger.connect(self.deepLearning_finish)
        t.step_trigger.connect(self.progress_step)
        t.start()

    def deepLearning_finish(self, result_path, band_order):
        """
        1、 异步深度学习提取信息后调用的方法
        Args:
            result_path: 带地理信息的tif图片地址
        Returns:
        """
        print("result_path:", result_path)
        self.band_order = band_order
        self.DeepLearning_Window.label = GuetShowLabel(self.DeepLearning_Window.widget)
        self.DeepLearning_Window.label.setGeometry(QtCore.QRect(0, 0, 1101, 731))
        self.DeepLearning_Window.label._graphyView.resize(QtCore.QSize(1101, 731))
        layout = QVBoxLayout()
        layout.addWidget(self.DeepLearning_Window.label)
        self.DeepLearning_Window.widget.setLayout(layout)
        self.img = GuetImage.get_rgbimgae_from_geographytif(result_path, band_order)
        self.DeepLearning_Window.show()
        self.DeepLearning_Window.label.add_img(self.img)
        self.DeepLearning_Window.pushButton.clicked.connect(self.save_file)

    def save_file(self):
        result = QFileDialog.getSaveFileName(self, '选择文件要保存的路径', './../result/DeepLearning_result/', "Image files (*.tif);; Image files (*.jpg *.png)")
        if result is not None:
            SavePath = QFileInfo(result[0]).path()
            SaveName = QFileInfo(result[0]).fileName()
            if SaveName == "":
                QMessageBox.critical(self, "错误", "文件保存失败")
            else:
                self.img.save(SavePath+'/{}'.format(SaveName))
                QMessageBox.information(self, "提示", "文件已保存")
        else:
            QFileDialog.exec_()

    def progress_step(self, total_step, current_step):
        print("progress {}/{}".format(current_step, total_step))
        # 动态添加进度条，算法运行过程中用于显示进度
        self.DeepLearning_Window.pgb.setRange(0, total_step)
        self.DeepLearning_Window.pgb.setValue(current_step)
        self.DeepLearning_Window.pgb.setFormat('提取进度为 %p%'.format(self.DeepLearning_Window.pgb.value()))
        if(total_step==current_step):
            self.DeepLearning_Window.pgb.deleteLater()

    def get_model_summary(self, tag, summary):
        self.tag = tag
        self.trainingmodel_summary = summary
        self.print_model_summary(tag, summary)

    def print_model_summary(self, tag, summary):
        self.show_trainingresult_window.cursot = self.show_trainingresult_window.textBrowser.textCursor()
        if tag == 1:
            self.show_trainingresult_window.cursot.select(QtGui.QTextCursor.LineUnderCursor)
            # 移除当前行内容
            self.show_trainingresult_window.cursot.removeSelectedText()
            # 移动光标到行首
            self.show_trainingresult_window.textBrowser.moveCursor(QtGui.QTextCursor.StartOfLine, QtGui.QTextCursor.MoveAnchor)
            # 重新设置值
            self.show_trainingresult_window.textBrowser.insertPlainText(summary)
        elif tag == 0:
            self.show_trainingresult_window.textBrowser.append(summary)  # 在指定的区域显示提示信息
            self.show_trainingresult_window.textBrowser.moveCursor(self.show_trainingresult_window.cursot.End)
        elif tag == 2:
            QMessageBox.information(self, "提示", "模型训练已结束，模型已保存到"+self.model_path)
        QtWidgets.QApplication.processEvents()


