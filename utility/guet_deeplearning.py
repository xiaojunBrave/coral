#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 使用CPU运行

from osgeo import gdal
import numpy as np
import cv2
import PIL.Image as Image


class GuetDeepLearning(object):
    # 读取tif数据集
    @staticmethod
    def readTif(fileName, xoff=0, yoff=0, data_width=0, data_height=0):
        dataset = gdal.Open(fileName)
        if dataset == None:
            print(fileName + "文件无法打开")
        # 栅格矩阵的列数
        width = dataset.RasterXSize
        # 栅格矩阵的行数
        height = dataset.RasterYSize
        # 波段数
        bands = dataset.RasterCount
        # 获取数据
        if (data_width == 0 and data_height == 0):
            data_width = width
            data_height = height
        data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
        # 获取仿射矩阵信息
        geotrans = dataset.GetGeoTransform()
        # 获取地图投影信息
        proj = dataset.GetProjection()
        return width, height, bands, data, geotrans, proj

    # 保存tif文件函数
    @staticmethod
    def writeTiff(im_data, im_geotrans, im_proj, path):
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        elif len(im_data.shape) == 2:
            im_data = np.array([im_data])
            im_bands, im_height, im_width = im_data.shape

        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
        if (dataset != None):
            dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
            dataset.SetProjection(im_proj)  # 写入投影
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
        del dataset

    # 线性拉伸
    @staticmethod
    def truncated_linear_stretch(image, truncated_value, max_out=255, min_out=0):
        def gray_process(gray):
            truncated_down = np.percentile(gray, truncated_value)
            truncated_up = np.percentile(gray, 100 - truncated_value)
            gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out
            np.clip(gray, min_out, max_out, out=gray)
            # gray[gray < min_out] = min_out
            # gray[gray > max_out] = max_out
            if (max_out <= 255):
                gray = np.uint8(gray)
            elif (max_out <= 65535):
                gray = np.uint16(gray)
            return gray
        #  如果是多波段
        if (len(image.shape) >= 3):
            image_stretch = []
            for i in range(image.shape[0]):
                gray = gray_process(image[i])
                image_stretch.append(gray)
            image_stretch = np.array(image_stretch)
        #  如果是单波段
        else:
            image_stretch = gray_process(image)
        return image_stretch

    # 对影像进行裁剪，将整幅影像裁剪为128×128大小的块，因为深度学习模型训练的时候就是使用的128×128大小的块，所以这里也是裁剪为128×128大小的块
    @staticmethod
    def TifCroppingArray(img, SideLength):
        # 裁剪链表
        TifArrayReturn = []
        # 列上图像块数目
        ColumnNum = int((img.shape[0] - SideLength * 2) / (128 - SideLength * 2))
        # 行上图像块数目
        RowNum = int((img.shape[1] - SideLength * 2) / (128 - SideLength * 2))
        for i in range(ColumnNum):
            TifArray = []
            for j in range(RowNum):
                cropped = img[i * (128 - SideLength * 2): i * (128 - SideLength * 2) + 128,
                          j * (128 - SideLength * 2): j * (128 - SideLength * 2) + 128]
                TifArray.append(cropped)
            TifArrayReturn.append(TifArray)
        # 考虑到行列会有剩余的情况，向前裁剪一行和一列
        # 向前裁剪最后一列
        for i in range(ColumnNum):
            cropped = img[i * (128 - SideLength * 2): i * (128 - SideLength * 2) + 128,
                      (img.shape[1] - 128): img.shape[1]]
            TifArrayReturn[i].append(cropped)
        # 向前裁剪最后一行
        TifArray = []
        for j in range(RowNum):
            cropped = img[(img.shape[0] - 128): img.shape[0],
                      j * (128 - SideLength * 2): j * (128 - SideLength * 2) + 128]
            TifArray.append(cropped)
        # 向前裁剪右下角
        cropped = img[(img.shape[0] - 128): img.shape[0],
                  (img.shape[1] - 128): img.shape[1]]
        TifArray.append(cropped)
        TifArrayReturn.append(TifArray)
        # 列上的剩余数
        ColumnOver = (img.shape[0] - SideLength * 2) % (128 - SideLength * 2) + SideLength
        # 行上的剩余数
        RowOver = (img.shape[1] - SideLength * 2) % (128 - SideLength * 2) + SideLength
        return TifArrayReturn, RowOver, ColumnOver

    # 标签可视化，即为第n类赋上n值
    @staticmethod
    def labelVisualize(img):
        img_out = np.zeros((img.shape[0], img.shape[1]))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                # 为第n类赋上n值
                img_out[i][j] = np.argmax(img[i][j])
        return img_out

    # 对测试图片进行归一化，并使其维度上和训练图片保持一致
    @staticmethod
    def testGenerator(TifArray):
        for i in range(len(TifArray)):
            for j in range(len(TifArray[0])):
                img = TifArray[i][j]
                # 归一化
                img = img / 255.0
                # 在不改变数据内容情况下，改变shape
                img = np.reshape(img, (1,) + img.shape)
                yield img

    # 将裁剪的块依次利用权重模型进行预测，并将预测按照原先影像进行拼接
    @staticmethod
    def Result(shape, TifArray, npyfile, RepetitiveLength, RowOver, ColumnOver):
        result = np.zeros(shape, np.uint8)

        # j来标记行数 item
        j = 0
        for i, item in enumerate(npyfile):
            img = GuetDeepLearning.labelVisualize(item)
            img = img.astype(np.uint8)

            # 最左侧一列特殊考虑，左边的边缘要拼接进去
            if (i % len(TifArray[0]) == 0):
                # 第一行的要再特殊考虑，上边的边缘要考虑进去
                if (j == 0):
                    result[0: 128 - RepetitiveLength, 0: 128 - RepetitiveLength] = img[0: 128 - RepetitiveLength,
                                                                                   0: 128 - RepetitiveLength]
                # 最后一行的要再特殊考虑，下边的边缘要考虑进去
                elif (j == len(TifArray) - 1):
                    result[shape[0] - ColumnOver - RepetitiveLength: shape[0], 0: 128 - RepetitiveLength] = img[
                                                                                                            128 - ColumnOver - RepetitiveLength: 128,
                                                                                                            0: 128 - RepetitiveLength]
                else:
                    result[j * (128 - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                                128 - 2 * RepetitiveLength) + RepetitiveLength,
                    0:128 - RepetitiveLength] = img[RepetitiveLength: 128 - RepetitiveLength, 0: 128 - RepetitiveLength]
            # 最右侧一列特殊考虑，右边的边缘要拼接进去
            elif (i % len(TifArray[0]) == len(TifArray[0]) - 1):
                # 第一行的要再特殊考虑，上边的边缘要考虑进去
                if (j == 0):
                    result[0: 128 - RepetitiveLength, shape[1] - RowOver: shape[1]] = img[0: 128 - RepetitiveLength,
                                                                                      128 - RowOver: 128]
                # 最后一行的要再特殊考虑，下边的边缘要考虑进去
                elif (j == len(TifArray) - 1):
                    result[shape[0] - ColumnOver: shape[0], shape[1] - RowOver: shape[1]] = img[128 - ColumnOver: 128,
                                                                                            128 - RowOver: 128]
                else:
                    result[j * (128 - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                                128 - 2 * RepetitiveLength) + RepetitiveLength,
                    shape[1] - RowOver: shape[1]] = img[RepetitiveLength: 128 - RepetitiveLength, 128 - RowOver: 128]
                # 走完每一行的最右侧，行数+1
                j = j + 1
            # 不是最左侧也不是最右侧的情况
            else:
                # 第一行的要特殊考虑，上边的边缘要考虑进去
                if (j == 0):
                    result[0: 128 - RepetitiveLength,
                    (i - j * len(TifArray[0])) * (128 - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                        TifArray[0]) + 1) * (128 - 2 * RepetitiveLength) + RepetitiveLength
                    ] = img[0: 128 - RepetitiveLength, RepetitiveLength: 128 - RepetitiveLength]
                # 最后一行的要特殊考虑，下边的边缘要考虑进去
                if (j == len(TifArray) - 1):
                    result[shape[0] - ColumnOver: shape[0],
                    (i - j * len(TifArray[0])) * (128 - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                        TifArray[0]) + 1) * (128 - 2 * RepetitiveLength) + RepetitiveLength
                    ] = img[128 - ColumnOver: 128, RepetitiveLength: 128 - RepetitiveLength]
                else:
                    result[j * (128 - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                                128 - 2 * RepetitiveLength) + RepetitiveLength,
                    (i - j * len(TifArray[0])) * (128 - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                        TifArray[0]) + 1) * (128 - 2 * RepetitiveLength) + RepetitiveLength,
                    ] = img[RepetitiveLength: 128 - RepetitiveLength, RepetitiveLength: 128 - RepetitiveLength]
        return result

    # 将单波段图像改为三波段
    @staticmethod
    def ThreeBand(ResultPath, savepath):

        image = Image.open(ResultPath)
        if len(image.split()) == 1:
            # 读原图片大小，创建个跟原图片尺寸相同的空图片
            img = cv2.imdecode(np.fromfile(ResultPath, dtype=np.uint8), cv2.IMREAD_COLOR)
            # 转为灰度图
            gray = cv2.imdecode(np.fromfile(ResultPath, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
            img2 = np.zeros_like(img)
            img2[:, :, 0] = gray
            img2[:, :, 1] = gray
            img2[:, :, 2] = gray
            cv2.imwrite(savepath, img2)
            image = Image.open(savepath)
            print(len(image.split()))
        else:
            image.save(savepath)

    # 改变图像像素值
    @staticmethod
    def RGB(result):
        img = Image.open(result).convert('RGB')
        print(img.size)  # 打印图片大小
        print(img.getpixel((4, 4)))
        width = img.size[0]  # 长度
        height = img.size[1]  # 宽度
        for i in range(0, width):  # 遍历所有长度的点
            for j in range(0, height):  # 遍历所有宽度的点
                data = (img.getpixel((i, j)))  # 打印该图片的所有点
                if (data[0] == 0 and data[1] == 0 and data[2] == 0):
                    img.putpixel((i, j), (255, 0, 0, 0))  # 类别为陆地
                if (data[0] == 1 and data[1] == 1 and data[2] == 1):
                    img.putpixel((i, j), (128, 128, 0, 0))  # 类别为藻类混合物
                if (data[0] == 2 and data[1] == 2 and data[2] == 2):
                    img.putpixel((i, j), (0, 255, 0, 0))  # 类别为健康珊瑚礁
                if (data[0] == 3 and data[1] == 3 and data[2] == 3):
                    img.putpixel((i, j), (255, 255, 0, 0))  # 类别为沙
                if (data[0] == 4 and data[1] == 4 and data[2] == 4):
                    img.putpixel((i, j), (0, 0, 255, 0))  # 类别为海水
                if (data[0] == 5 and data[1] == 5 and data[2] == 5):
                    img.putpixel((i, j), (255, 0, 255, 0))  # 类别为白化珊瑚礁
                if (data[0] == 6 and data[1] == 6 and data[2] == 6):
                    img.putpixel((i, j), (0, 255, 255, 0))  # 类别为破浪
        img = img.convert("RGB")  # 把图片强制转成RGB
        img.save(result)  # 保存修改像素点后的图片