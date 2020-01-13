# 图像直方图分布统计的是像素值大小的分布情况，0-255，每一个亮度大小的取值情况，直方图分布其实就可以看做是亮度分布。
# 直方图增强主要作用是用来处理较暗或者较亮的情况，让图像的整体亮度分布均匀。
# 此外图像的对比度是通过灰度级范围来度量的，而灰度级范围可通过观察灰度直方图得到，灰度级范围越大代表对比度越高；反之对比度越低，低对比度的图像在视觉上给人的感觉是看起来不够清晰

#直方图计算numpy和opencv都可以
#参考连接：https://blog.csdn.net/v_xchen_v/article/details/79913245，

# 彩色图像均衡，就是拆分成多个通道分别均衡再组合起来。

import cv2
import  numpy as np
import math
import logging

# # 伽马变换

# img = cv2.imread("test.jpg", 0)
# # 图像归一化
# fi = img / 255.0
# # 伽马变换
# gamma = 1.4
# out = np.power(fi, gamma)
# cv2.imshow("img", img)
# cv2.imshow("out", out)
# cv2.waitKey()
#
#
# #全局直方图均衡
# def equalHist(img):
#     # 灰度图像矩阵的高、宽
#     h, w = img.shape
#     # 第一步：计算灰度直方图
#     grayHist = cv2.calcHist(img)(img)
#     # 第二步：计算累加灰度直方图
#     zeroCumuMoment = np.zeros([256], np.uint32)
#     for p in range(256):
#         if p == 0:
#             zeroCumuMoment[p] = grayHist[0]
#         else:
#             zeroCumuMoment[p] = zeroCumuMoment[p - 1] + grayHist[p]
#     # 第三步：根据累加灰度直方图得到输入灰度级和输出灰度级之间的映射关系
#     outPut_q = np.zeros([256], np.uint8)
#     cofficient = 256.0 / (h * w)
#     for p in range(256):
#         q = cofficient * float(zeroCumuMoment[p]) - 1
#         if q >= 0:
#             outPut_q[p] = math.floor(q)
#         else:
#             outPut_q[p] = 0
#     # 第四步：得到直方图均衡化后的图像
#     equalHistImage = np.zeros(img.shape, np.uint8)
#     for i in range(h):
#         for j in range(w):
#             equalHistImage[i][j] = outPut_q[img[i][j]]
#     return equalHistImage
#
# img = cv2.imread("test.jpg", 0)
# if img is not None:
#     logging.debug("img is None")
#
# # 使用自己写的函数实现
# # equa = equalHist(img)
# # grayHist(img, equa)
#
# # 使用OpenCV提供的直方图均衡化函数实现
# equa = cv2.equalizeHist(img)
# cv2.imshow("img", img)
# cv2.imshow("equa", equa)
# cv2.waitKey()


# 自适应直方图均衡

img = cv2.imread("1060-4_rlt.png", 0)
if img is not None:
    logging.debug("img is None")

def auto_histogram_equalization(imgChannal):

    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值均衡化
    dst = clahe.apply(img)
    cv2.imshow("img", img)
    cv2.imshow("dst", dst)
    cv2.waitKey()

def mutil_channal_equalization(img,is_singal_channal=True):

    # 分离图像
    channal=[]
    if is_singal_channal:
        channal_num=1
        channal=channal+img
    else:
        img_shape=img.shape
        if len(img_shape)==2:
            logging.debug("img is singal channal")
        else:
            channal_num=img_shape[2]
            for i in range(channal_num):
                channal=channal+img[i]

    # 直方图均衡




