import cv2
import numpy as np


def preprocess(gray):
    # 1. Sobel算子，x方向求梯度
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # 2. 二值化
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # 3. 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 15))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))
    element3 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 9))
    # 4. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)
    # cv2.imshow("kajh",dilation)

    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    erosion = cv2.erode(dilation, element1, iterations=1)
    # cv2.imshow("a",erosion)

    # 6. 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element3, iterations=3)
    # cv2.imshow("b",dilation2)
    # cv2.waitKey()

    # # 7. 存储中间图片
    # cv2.imwrite("binary.png", binary)
    # cv2.imwrite("dilation.png", dilation)
    # cv2.imwrite("erosion.png", erosion)
    # cv2.imwrite("dilation2.png", dilation2)

    return dilation2


def findTextRegion(img):
    region = []
    shape=img.shape
    img_center_high=shape[0]/2
    img_center_width=shape[1]/2
    # 1. 查找轮廓
    _,contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)

        # 面积小的都筛选掉
        if (area < 1000):
            continue

        # 轮廓近似，作用很小
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)
        # print("rect is", rect)

        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 筛选那些太细的矩形，留下扁的
        if (height > width * 1.5):
            continue
        # 删除那些太短的，但是被错误包括在内的数据
        if width < 100:
            continue

        # 删除边缘框
        right = box[2][0] + 0
        left = box[0][0] - 0
        high = box[0][1] + 0
        low = box[2][1] - 0
        box_center_high=abs((low+high)/2)
        box_cnter_width=abs((right+left)/2)
        high_sub=abs(box_center_high-img_center_high)
        width_sub=abs(box_cnter_width-img_center_width)
        # print("box",high_sub,width_sub)
        if high_sub+width_sub>400:
            continue
        region.append(box)

    return region
