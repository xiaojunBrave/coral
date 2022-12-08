#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CoralReefRecognitionProject 
@File ：guet_trian.py
@IDE  ：PyCharm 
@Author ：xiaoj
@Date ：2022/12/2 20:17 
"""
import os
import sys

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QWidget

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from Generator import trainGenerator, valGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import cv2
from utility.model import unet
import keras.callbacks


class TrainCustomCallBack(keras.callbacks.Callback):
    def __init__(self, epochs, batch_size, img_size, model, call_back):
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.batchs = img_size / batch_size
        self.model = model
        self.call_back = call_back

    def on_train_begin(self, logs=None):
        pass
        #print("xiaoj on_train_begin")

    def on_train_end(self, logs=None):
        pass
        # print("xiaoj on_train_end")

    def on_train_batch_begin(self, batch, logs=None):
        pass
        # print("on_train_batch_begin")

    def on_train_batch_end(self, batch, logs=None):
        self.call_back(0 if batch == 0 else 1, "%d/%d %s -loss:%.4f -accuracy:%.4f" % (batch + 1, self.batchs, self.get_progress(batch + 1),
                                                                                       logs["loss"], logs["accuracy"]))

    def on_epoch_end(self, epoch, logs=None):
        self.call_back(0,
                       "%d/%d %s -loss:%.4f -accuracy:%.4f -val_loss:%.4f -val_accuracy:%.4f -lr:%.5f" % (self.batchs, self.batchs, self.get_progress(self.batchs),
                                                               logs["loss"], logs["accuracy"], logs["val_loss"], logs["val_accuracy"], logs["lr"]))

    def on_epoch_begin(self, epoch, logs=None):
        self.call_back(0, "Epoch {}/{}".format(epoch+1,self.epochs))
        # print("on_epoch_begin")

    def get_progress(self,batch):
        s = "["
        index = batch / self.batchs * 30
        for i in range(30):
            if i <= index:
                s += ">"
            else:
                s += "."
        s += "]"
        return s


class GuetTrainThread(QThread):
    model_summary = pyqtSignal(int, str)
    step_trigger = pyqtSignal(int, int)

    def __init__(self,**kw):
        super(GuetTrainThread, self).__init__()
        self.model_path = kw["model_path"]
        self.train_image_path = kw["train_image_path"]
        self.train_label_path = kw["train_label_path"]
        self.validation_image_path = kw["validation_image_path"]
        self.validation_label_path = kw["validation_label_path"]
        self.batch_size = kw["batch_size"]
        self.classNum = kw["classNum"]
        self.input_size = kw["input_size"]
        self.epochs = kw["epochs"]
        self.learning_rate = kw["learning_rate"]
        self.premodel_path = kw["premodel_path"]

    def run(self) -> None:
        # 获取文件夹内的文件名，train_label_path为上方已经赋值的标签路径
        ImageNameList = os.listdir(self.train_label_path)
        # 训练数据数目
        train_num = len(os.listdir(self.train_image_path))
        # 验证数据数目
        validation_num = len(os.listdir(self.validation_image_path))
        # 训练集每个epoch有多少个batch_size
        steps_per_epoch = train_num / self.batch_size
        # 验证集每个epoch有多少个batch_size
        validation_steps = validation_num / self.batch_size
        # 提取标签颜色的缓存字典
        buff1 = []
        buff2 = []

        for i in range(len(ImageNameList)):
            ImagePath = self.train_label_path + "/" + ImageNameList[i]
            img0 = cv2.imdecode(np.fromfile(ImagePath, dtype=np.uint8), cv2.IMREAD_COLOR).astype(np.uint32)

            #img0 = cv2.imread(ImagePath).astype(np.uint32)
            # 为了提取唯一值，将RGB转成一个数
            img1 = img0[:, :, 0] * 1000000 + img0[:, :, 1] * 1000 + img0[:, :, 2]
            img2 = np.unique(img1)
            # 将像素矩阵的值添加到缓冲里
            for j in range(img2.shape[0]):
                buff1.append(img2[j])
            # 对目前i个像素矩阵里的唯一值再取唯一值
            buff1 = sorted(set(buff1))
            # 当目前存储的类型达到类别总数时，停止遍历剩余的图像
            if (len(buff1) == self.classNum):
                break

        for k in range(self.classNum):
            color = str(buff1[k]).rjust(9, '0')
            # 提取RGB值
            buffRGB = [int(color[0: 3]), int(color[3: 6]), int(color[6: 9])]
            buff2.append(buffRGB)

        # 转为numpy格式
        buff2 = np.array(buff2)
        # 存储灰度颜色字典
        buff3 = buff2.reshape((self.classNum, 1, 3)).astype(np.uint8)
        buff3 = cv2.cvtColor(buff3, cv2.COLOR_BGR2GRAY)

        # 得到一个生成器，以batch_size的速率生成训练数据
        train_Generator = trainGenerator(self.batch_size,
                                         self.train_image_path,
                                         self.train_label_path,
                                         self.classNum,
                                         buff3,
                                         self.input_size
                                         )

        # 得到一个生成器，以batch_size的速率生成验证数据
        validation_data = valGenerator(self.batch_size,
                                       self.validation_image_path,
                                       self.validation_label_path,
                                       self.classNum,
                                       buff3,
                                       self.input_size
                                       )

        # 调用模型文件
        model = unet(pretrained_weights=self.premodel_path,
                     input_size=self.input_size,
                     classNum=self.classNum,
                     learning_rate=self.learning_rate)

        # 输出模型结构
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        self.model_summary.emit(0, short_model_summary)
        # 回调函数
        # val_loss连续10轮没有下降则停止训练
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        # 当3个epoch过去而val_loss不下降时，学习率减半
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        model_checkpoint = ModelCheckpoint(self.model_path,
                                           monitor='loss',
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=True)
        custTrain = TrainCustomCallBack(self.epochs, self.batch_size, len(ImageNameList), model, self.send_message)
        # 模型训练
        history = model.fit_generator(train_Generator,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=self.epochs,
                                      callbacks=[early_stopping, model_checkpoint, reduce_lr, custTrain],
                                      validation_data=validation_data,
                                      validation_steps=validation_steps)
        self.model_summary.emit(2, "")

    def send_message(self, tag, message):
        self.model_summary.emit(tag, message)

class GuetTrain(QWidget):
    model_summary = pyqtSignal(int, str)
    def train(self, model_path, train_image_path, train_label_path, validation_image_path, validation_label_path,
              batch_size, classNum, input_size, epochs, learning_rate, premodel_path):
        # print("model_path", model_path)
        # print("train_image_path", train_image_path)
        # print("train_label_path", train_label_path)
        # print("validation_image_path", validation_image_path)
        # print("validation_label_path", validation_label_path)
        # print("batch_size", batch_size)
        # print("classNum", classNum)
        # print("input_size", input_size)
        # print("epochs", epochs)
        # print("learning_rate", learning_rate)
        # print("premodel_path", premodel_path)
        # model_path = "E:/study/研究生/中科院项目/海南珊瑚礁底栖物质识别平台/Ui_table/utility/weight/test.hdf5"
        # train_image_path = "C:/Users/xukejian/Desktop/训练模块代码/data/train/image"
        # train_label_path = "C:/Users/xukejian/Desktop/训练模块代码/data/train/label"
        # validation_image_path = "C:/Users/xukejian/Desktop/训练模块代码/data/val/image"
        # validation_label_path = "C:/Users/xukejian/Desktop/训练模块代码/data/val/label"
        # batch_size = 2
        # classNum = 7
        # input_size = (128, 128, 3)
        # epochs = 2
        # learning_rate = 0.0001
        # premodel_path = None
        """
        模型训练
        Args:
            model_path:训练模型保存路径
            train_image_path:训练数据路径
            train_label_path:训练标签路径
            validation_image_path:验证数据路径
            validation_label_path:验证标签路径
            batch_size:批大小
            classNum:类别的数目
            input_size:输入图像大小
            epochs:训练模型的迭代总轮数
            learning_rate:初始学习率
            premodel_path:是否有预训练模型？有的话输入路径及文件名，如"./Model/unet_model.hdf5"，如果没有则默认None

        Returns:

        """


        t = GuetTrainThread(model_path=model_path,train_image_path=train_image_path,train_label_path=train_label_path,
                            validation_image_path=validation_image_path,validation_label_path=validation_label_path,
                            batch_size=batch_size,classNum=classNum,input_size=input_size,epochs=epochs,
                            learning_rate=learning_rate,premodel_path=premodel_path)
        t.model_summary.connect(self.show_model_summary)
        #t.step_trigger.connect(self.progress_step)
        t.start()

    def show_model_summary(self, tag, summary):
        self.model_summary.emit(tag, summary)
