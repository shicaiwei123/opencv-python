import cv2
import numpy as np


def usm_enchace(img, num=1):
    '''
    利用usm方式对图像进行增强
    https://zhuanlan.zhihu.com/p/63502539
    :param img:
    :return:
    '''
    # sigma = 5、15、25
    for i in range(num):
        blur_img = cv2.GaussianBlur(img, (0, 0), 5)
        usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
        img = usm

    return img


def usm_test():
    '''
    usm的测试程序
    :return:
    '''

    path = "/home/shicaiwei/project/opencv-python/data/6/1722-0.jpg"
    src = cv2.imread(path)
    usm = usm_enchace(src)

    cv2.namedWindow("input", 0)
    cv2.imshow("input", src)

    cv2.namedWindow("mask image", 0)
    cv2.imshow("mask image", usm)

    # 拼接显示
    h, w = src.shape[:2]
    result = np.zeros([h, w * 2, 3], dtype=src.dtype)
    result[0:h, 0:w, :] = src
    result[0:h, w:2 * w, :] = usm

    cv2.namedWindow("sharpen_image", 0)
    cv2.imshow("sharpen_image", result)
    cv2.imwrite("./result.png", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    usm_test()
