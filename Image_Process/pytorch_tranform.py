import random
import numpy as np
import cv2


def random_crop_and_resize(image, size=224):
    '''
    将一张图片裁剪出随机大小的224×224大小的正方形区域。
    :param image:
    :param size:
    :return:
    '''

    h, w = image.shape[:2]

    y = np.random.randint(0, h - size)
    x = np.random.randint(0, w - size)
    print(x, y)

    image = image[y:y + size, x:x + size, :]

    return image


def rotate_bound(image, angle):
    '''
    随机在正负angle范围内旋转图像
    :param image:
    :param angle:
    :return:
    '''
    # 获取图像的尺寸
    # 旋转中心
    (h, w) = image.shape[:2]
    (cx, cy) = (w / 2, h / 2)

    # 设置旋转矩阵
    angle = random.randint(-angle, angle)
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos = np.abs(M[0, 0]) * 0.8
    sin = np.abs(M[0, 1]) * 0.8

    # 计算图像旋转后的新边界
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))

    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0, 2] += (nw / 2) - cx
    M[1, 2] += (nh / 2) - cy

    return cv2.warpAffine(image, M, (nw, nh))


if __name__ == '__main__':
    img_path = "/home/shicaiwei/data/liveness_data/CASIA-SUFR/Training/real_part/CLKJ_AS0137/real.rssdk/depth/31.jpg"
    img = cv2.imread(img_path)
    img = cv2.resize(img, (144, 144))
    img_random_crop = rotate_bound(img, 30)
    cv2.imshow("imh_rondom", img_random_crop)
    cv2.waitKey(0)
