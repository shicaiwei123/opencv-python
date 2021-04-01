import cv2
import numpy as np


def dog_of_img(img, filter_num, display=False):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 参数来自sift算子
    sigma = 1.6
    k = 2 ** (filter_num / 2)
    filter_result = []
    filter_result.append(img_gray)
    for i in range(filter_num):
        filtered_img = cv2.GaussianBlur(img_gray, (3, 3), sigmaX=0)
        if display:
            cv2.imshow("img", filtered_img)
            cv2.waitKey(0)
        filter_result.append(filtered_img)

    dog_result = []
    for index in range(len(filter_result) - 1):
        dog_img = np.float32(filter_result[index]) - np.float32(filter_result[index + 1])
        dog_img = ((dog_img - np.min(dog_img)) / (np.max(dog_img) - np.min(dog_img))) * 255
        dog_img = np.uint8(dog_img)
        if display:
            cv2.imshow("img_dog", dog_img)
            cv2.waitKey(0)
        dog_result.append(dog_img)
    dog_result.append(filter_result[filter_num-1])

    print(1)


# 转float?

if __name__ == '__main__':
    img_path = "/home/shicaiwei/data/liveness_data/cross_photo_face_normal/train/living/3/34.jpg"
    img = cv2.imread(img_path)
    dog_of_img(img=img, filter_num=2, display=True)
