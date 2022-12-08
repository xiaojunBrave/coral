import cv2
import numpy as np
from PIL import Image
from numpy import uint8

from utility.guet_image import GuetImage


class GuetBrightnessAdjustment(object):

    # @staticmethod
    # def RGBAlgorithm(rgb_img, value):
    #     img = rgb_img * 1.0
    #     # img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    #     img_out = img
    #     # 基于当前RGB进行调整（RGB*alpha）
    #     # 增量大于0，指数调整
    #     if value >= 0 and value != 1:
    #         alpha = 1 - value
    #         alpha = 1 / alpha
    #         img_out[:, :, 0] = img[:, :, 0] * alpha
    #         img_out[:, :, 1] = img[:, :, 1] * alpha
    #         img_out[:, :, 2] = img[:, :, 2] * alpha
    #
    #     # 增量小于0，线性调整
    #     elif value < 0:
    #         alpha = value + 1
    #         img_out[:, :, 0] = img[:, :, 0] * alpha
    #         img_out[:, :, 1] = img[:, :, 1] * alpha
    #         img_out[:, :, 2] = img[:, :, 2] * alpha
    #
    #     # 独立于当前RGB进行调整（RGB+alpha*255）
    #     else:
    #         alpha = value
    #         img_out[:, :, 0] = img[:, :, 0] + 255.0 * alpha
    #         img_out[:, :, 1] = img[:, :, 1] + 255.0 * alpha
    #         img_out[:, :, 2] = img[:, :, 2] + 255.0 * alpha
    #
    #     img_out = img_out / 255.0
    #
    #     # RGB颜色上下限处理(小于0取0，大于1取1)
    #     mask_3 = img_out < 0
    #     mask_4 = img_out > 1
    #     img_out = img_out * (1 - mask_3)
    #     img_out = img_out * (1 - mask_4) + mask_4
    #     return img_out


    # @staticmethod
    # def RGBAlgorithm(file_path, value):
    #     cnum = 10
    #     img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    #     img_rgb = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #     for i in range(img.shape[0]):
    #         for j in range(img.shape[1]):
    #             lst = 0.1 * cnum * img[i, j] + value
    #             img_rgb[i, j] = [int(ele) if ele < 255 else 255 for ele in lst]
    #     return img_rgb


    @staticmethod
    def RGBAlgorithm(data, value):
        cnum = 10
        # img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        # img_rgb = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        # for i in range(img.shape[0]):
        #     for j in range(img.shape[1]):
        #         lst = 0.1 * cnum * img[i, j] + value
        #         img_rgb[i, j] = [int(ele) if ele < 255 else 255 for ele in lst]
        img_data = data.copy()
        print("value", value)
        im_bands, im_height, im_width = data.shape
        # img = GuetImage._tig_to_jpg(data)
        for i in range(im_bands):
            # av = np.median(data[:,:,i])
            # data[:,:,i] += uint8(av * uint8(value))
            minValue = 0
            maxValue = 255
            # 获取数组RGB_Array某个百分比分位上的值
            low_value = np.percentile(data[:, :, i], 0.5)
            high_value = np.percentile(data[:, :, i], 99.5)
            temp_value = (data[:, :, i] - low_value) * maxValue / (high_value - low_value) + value
            temp_value[temp_value < minValue] = minValue
            temp_value[temp_value > maxValue] = maxValue
            data[:, :, i] = temp_value
        return data

