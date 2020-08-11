# 图像直方图分布统计的是像素值大小的分布情况，0-255，每一个亮度大小的取值情况，直方图分布其实就可以看做是亮度分布。
# 直方图增强主要作用是用来处理较暗或者较亮的情况，让图像的整体亮度分布均匀。
# 此外图像的对比度是通过灰度级范围来度量的，而灰度级范围可通过观察灰度直方图得到，灰度级范围越大代表对比度越高；反之对比度越低，低对比度的图像在视觉上给人的感觉是看起来不够清晰

# 直方图计算numpy和opencv都可以
# 参考连接：https://blog.csdn.net/v_xchen_v/article/details/79913245，

# 彩色图像均衡，就是拆分成多个通道分别均衡再组合起来。

import cv2
import numpy as np
import math
import logging


# # 伽马变换
def gamma_adjust(im, gamma=1.0,is_hsv=False):
    """伽马矫正"""

    if is_hsv:
        img_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)

        v_gamma =( np.power(v.astype(np.float32) / 255, 1 / gamma) * 255).astype(np.uint8)
        img_hsv_gamma=cv2.merge((h,s,v_gamma))
        img_gamma=cv2.cvtColor(img_hsv_gamma,cv2.COLOR_HSV2BGR)
    else:
        img_gamma=( np.power(im.astype(np.float32) / 255, 1 / gamma) * 255).astype(np.uint8)
    return img_gamma


def CLAHE_signle(img):
    '''
    单通道限制对比度直方图均衡
    :param img:
    :return:
    '''
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值均衡化
    dst = clahe.apply(img)
    return dst


# 自适应直方图均衡
def CLAHE_multi(img):
    '''
    多通道限制对比度直方图均衡
    :param img:
    :return:
    '''
    # 限制对比度的自适应阈值均衡化
    B, G, R = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    B = clahe.apply(B)
    G = clahe.apply(G)
    R = clahe.apply(R)
    img_hist = cv2.merge([B, G, R])

    return img_hist


def global_he_multi(img):
    '''
    彩色图像的全局直方图均衡
    :param img:
    :return:
    '''

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    v_he = cv2.equalizeHist(v)

    img_hsv_he = cv2.merge((h, s, v_he))

    img_he = cv2.cvtColor(img_hsv_he, cv2.COLOR_HSV2BGR)

    return img_he


def he_test():
    '''
    不同直方图均衡方法的测试
    :return:
    '''
    path = "/home/shicaiwei/project/opencv-python/data/6/1722-0.jpg"
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img is not None:
        logging.debug("img is None")

    # 使用OpenCV提供的直方图均衡化函数实现
    equa_opencv = global_he_multi(img)

    # single_CLAHE
    equa_CLAHE_single = CLAHE_signle(img_gray)

    # multi_CLAHE
    equa_CLAHE_multi = CLAHE_multi(img)

    # gamma校正
    img_gamma = gamma_adjust(img, gamma=1.5)

    # gamma 校正 hsv
    img_gamma_hsv=gamma_adjust(img,gamma=1.5,is_hsv=True)

    cv2.namedWindow("img", 0)
    cv2.namedWindow("equa_opencv", 0)
    cv2.namedWindow("equa_CLAHE_single", 0)
    cv2.namedWindow("equa_CLAHE_multi", 0)

    cv2.imshow("img", img)
    cv2.imshow("equa_opencv", equa_opencv)
    cv2.imshow("equa_CLAHE_single", equa_CLAHE_single)
    cv2.imshow("equa_CLAHE_multi", equa_CLAHE_multi)

    cv2.namedWindow("img_gamma", 0)
    cv2.imshow("img_gamma", img_gamma)

    cv2.namedWindow("img_gamma_hsv", 0)
    cv2.imshow("img_gamma_hsv", img_gamma_hsv)

    cv2.waitKey()


if __name__ == '__main__':
    he_test()

