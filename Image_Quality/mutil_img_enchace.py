import cv2
from utils import get_file_list
import numpy as np
from retinax import retianx_process

def add_img(img1, img2):
    '''
    两张图片相加
    :param img1:
    :param img2:
    :return:
    '''
    img1_shape = img1.shape
    x = img1_shape[0]
    y = img1_shape[1]
    img2 = cv2.resize(img2, (y, x))

    img1 = np.float32(img1)
    img2 = np.float32(img2)

    img_add = (img1 + img2) / 2
    img_add = np.uint8(img_add)
    return img_add


def sub_img(img1, img2):
    '''
    两张图片减
    :param img1:
    :param img2:
    :return:
    '''
    img1_shape = img1.shape
    x = img1_shape[0]
    y = img1_shape[1]
    img2 = cv2.resize(img2, (y, x))

    img1 = np.float32(img1)
    img2 = np.float32(img2)

    img_add = (img1 - img2)
    return img_add


def ying_iccv_2017(img):
    '''
    图像增强
    https://github.com/baidut/OpenCE
    :param img:
    :return:
    '''

    # 参数
    alpha = 1
    eps = 0.001
    range = 5
    ratioMax = 7
    img = np.float32(img)
    img = img / 255

    def perffer(img, k):
        a = 4.3536
        b = 1.2854
        c = 0.1447

        img = (img * k ** (a * c)) / (((img) ** (1 / c) * (k ** a - 1) + 1) ** c)
        img = (img * 255)



def img_enchace(img_path):
    '''
    读取文件夹的图片,使用各类方法增强图片
    :param img_path:
    :return:
    '''

    file_path_list = get_file_list(img_path)
    count = 1

    # 相邻照片求噪声
    img1 = cv2.imread("/home/shicaiwei/project/opencv-python/data/6/1721-0.jpg")
    img2 = cv2.imread("/home/shicaiwei/project/opencv-python/data/6/1722-0.jpg")
    img_noise = sub_img(img1, img2)

    # 叠加
    img_add = cv2.imread(file_path_list[0])
    for i in range(1, len(file_path_list)):
        img = cv2.imread(file_path_list[i])
        img_sub_noise = sub_img(img, img_noise)
        img_add = add_img(img_add, img)
        count += 1

    B, G, R = cv2.split(img_add)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    B = clahe.apply(B)
    G = clahe.apply(G)
    R = clahe.apply(R)
    img_add_hist = cv2.merge([B, G, R])

    img_add_add_hist = add_img(img_add, img_add_hist)

    img_msr = retianx_process(img_add)
    img_add_add_msr = add_img(img_add, img_msr)

    cv2.namedWindow("img", 0)
    cv2.namedWindow("img_add", 0)
    cv2.namedWindow("img_add_hist", 0)
    cv2.namedWindow("img_add_add_hist", 0)
    cv2.namedWindow("img_msr", 0)
    cv2.namedWindow("img_add_add_msr", 0)

    cv2.imshow("img", img)
    cv2.imshow("img_add", img_add)
    cv2.imshow("img_add_hist", img_add_hist)
    cv2.imshow("img_add_add_hist", img_add_add_hist)
    cv2.imshow("img_msr", img_msr)
    cv2.imshow("img_add_add_msr", img_add_add_msr)
    cv2.waitKey(0)


if __name__ == '__main__':
    img_path = "/home/shicaiwei/project/opencv-python/data/6"
    img_enchace(img_path)

