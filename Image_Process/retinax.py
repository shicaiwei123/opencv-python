import cv2
import numpy as np


def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


def SSR(src_img, size):
    '''
    https://blog.csdn.net/wsp_1138886114/article/details/83096109
    :param src_img:
    :param size:
    :return:
    '''
    L_blur = cv2.GaussianBlur(src_img, (size, size), 0)
    img = replaceZeroes(src_img)
    L_blur = replaceZeroes(L_blur)

    dst_Img = cv2.log(img / 255.0)
    dst_Lblur = cv2.log(L_blur / 255.0)
    dst_IxL = cv2.multiply(dst_Img, dst_Lblur)
    log_R = cv2.subtract(dst_Img, dst_IxL)

    dst_R = cv2.normalize(log_R, None, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8


def my_ssr(src_img, size):
    '''
    按照定义书写的ssr,但是效果不好,有待解决
    :param src_img:
    :param size:
    :return:
    '''
    # 求光照分量
    L_blur = cv2.GaussianBlur(src_img, (size, size), 0)
    L_blur = np.float32(L_blur)
    L_blur_min = np.min(L_blur)
    L_blur = (L_blur - L_blur_min)

    # 求反射分类
    src_img = src_img / 255.0
    L_blur = L_blur / 255.0
    src_img = np.float32(src_img)
    log_blur = cv2.log(L_blur + 1.0)
    log_img = cv2.log(src_img + 1.0)

    log_r = log_img - log_blur


    ssr = (log_r - np.min(log_r)) / (np.max(log_r) - np.min(log_r))
    ssr = np.uint8(ssr * 255)
    return ssr


def MSR(img, scales):
    '''
    https://blog.csdn.net/wsp_1138886114/article/details/83096109
    :param img:
    :param scales:
    :return:
    '''
    weight = 1 / 3.0
    scales_size = len(scales)
    h, w = img.shape[:2]
    log_R = np.zeros((h, w), dtype=np.float32)

    for i in range(scales_size):
        img = replaceZeroes(img)
        L_blur = cv2.GaussianBlur(img, (scales[i], scales[i]), 0)
        L_blur = replaceZeroes(L_blur)
        dst_Img = cv2.log(img / 255.0)
        dst_Lblur = cv2.log(L_blur / 255.0)
        dst_Ixl = cv2.multiply(dst_Img, dst_Lblur)
        log_R += weight * cv2.subtract(dst_Img, dst_Ixl)

    dst_R = cv2.normalize(log_R, None, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8


def retianx_process(img, mode='MSR'):
    if mode == 'SSR':
        scales = 3  # 奇数
        src_img = img
        img_hsv = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(img_hsv)
        V_retianx = SSR(V, scales)
        img_retianx = cv2.merge([H, S, V_retianx])
        img_result = cv2.cvtColor(img_retianx, cv2.COLOR_HSV2BGR)
        return img_result

    elif mode == 'MSR':

        scales = [3, 5, 9]  # 奇数

        src_img = img
        img_hsv = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(img_hsv)

        V_retianx = MSR(V, scales)
        img_retianx = cv2.merge([H, S, V_retianx])
        img_result = cv2.cvtColor(img_retianx, cv2.COLOR_HSV2BGR)
        return img_result

    else:
        print("error mode")


def retianx_test():
    '''
    SSR和MSR的测试代码
    :return:
    '''
    path = "/home/shicaiwei/data/liveness_data/cross_paper_face_normal/train/living/1/34.jpg"

    scales = 81  # 奇数

    src_img = cv2.imread(path)
    H_src, S, V_src = cv2.split(src_img)

    ssr_result = retianx_process(src_img, mode='SSR')
    msr_result = retianx_process(src_img, mode='MSR')

    H_ssr, S, V_ssr = cv2.split(ssr_result)
    H_msr, S, V_msr = cv2.split(msr_result)

    V_src_mean = np.mean(V_src)
    print("V_src_mean", V_src_mean)

    V_ssr_mean = np.mean(V_ssr)
    print("V_ssr_mean", V_ssr_mean)

    V_msr_mean = np.mean(V_msr)
    print("V_msr_mean", V_msr_mean)

    cv2.namedWindow("H_src", 0)
    H_src[H_src < 100] = 0
    H_src[H_src > 150] = 0
    cv2.imshow("H_src", H_src)

    cv2.namedWindow("H_ssr", 0)
    H_ssr[H_ssr < 170] = 0
    H_ssr[H_ssr > 220] = 0
    cv2.imshow("H_ssr", H_ssr)

    cv2.namedWindow("H_msr", 0)
    cv2.imshow("H_msr", H_msr)

    cv2.namedWindow("img", 0)
    cv2.namedWindow("SSR_result", 0)
    cv2.namedWindow("MSR_result", 0)
    cv2.imshow('img', src_img)
    cv2.imshow('SSR_result', ssr_result)
    cv2.imshow('MSR_result', msr_result)
    cv2.waitKey(0)


if __name__ == '__main__':
    retianx_test()
