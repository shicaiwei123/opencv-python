import cv2
from Image_Process.guide_filter import GuidedFilter, FastGuidedFilter
import datetime


def guide_filter(img, display):
    # load your image
    radius = 2
    eps = 0.001
    GF = FastGuidedFilter(img, radius, eps,scale=1)
    time_begin = datetime.datetime.now()
    img_blur = GF.filter(img)

    time_end = datetime.datetime.now()
    time_all = time_end - time_begin

    print("time guide filter", time_all.total_seconds())
    if display:
        cv2.namedWindow("guide_filter", 0)
        cv2.imshow("guide_filter", img_blur)


def bilateral_filter(img, display):
    time_begin = datetime.datetime.now()

    img_blur = cv2.bilateralFilter(img, 3, 75, 75)

    time_end = datetime.datetime.now()
    time_all = time_end - time_begin

    print("bilateral_filter", time_all.total_seconds())

    if display:
        cv2.namedWindow("bilateralFilter", 0)
        cv2.imshow("bilateralFilter", img_blur)


if __name__ == '__main__':
    img_path = "/home/shicaiwei/project/opencv-python/data/edge_preserve/1.png"
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    cv2.namedWindow("origin", 0)
    cv2.imshow("origin", img)

    bilateral_filter(img, display=True)

    guide_filter(img, display=True)

    cv2.waitKey(0)
