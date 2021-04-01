import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def fre_phase_analysis(img, display=False):
    '''
    获取并显示图像的频谱图和相位图
    :param img:
    :param display:
    :return:
    '''
    time_begin = time.time()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    phase_spectrum = np.angle(fshift)
    divide_num = np.max(phase_spectrum) - np.min(phase_spectrum)
    if divide_num == 0:
        divide_num = 6.283
    phase_spectrum_normal = np.uint8(255 * (phase_spectrum - np.min(phase_spectrum)) / divide_num)
    time_end = time.time()
    print('time', time_end - time_begin)

    # 取绝对值：将复数求模值
    # 取对数的目的为了将数据变化到较小的范围（比如0-255）
    fre = np.abs(fshift)

    s1 = np.uint8(np.log(np.abs(f) + 1))
    s2 = np.uint8(np.log(np.abs(fshift) + 1))
    s2 = ((s2 - np.min(s2)) / (np.max(s2) -np.min(s2)))*255
    plt.subplot(221), plt.imshow(img_gray, cmap='gray'), plt.title('original img')
    plt.subplot(222), plt.imshow(s1, cmap='gray'), plt.title('original fre')
    plt.subplot(223), plt.imshow(s2, cmap='gray'), plt.title('center fre')
    plt.subplot(224), plt.imshow(phase_spectrum_normal, cmap='gray'), plt.title('phase')
    plt.show()


def componenet_analysis(img, display=False):
    img_blur = cv2.GaussianBlur(img, ksize=(3, 3),sigmaX=0)
    # img_blur = img
    fre_phase_analysis(img_blur, display=display)
    img_hpf = img - img_blur
    fre_phase_analysis(img_hpf, display=display)


if __name__ == '__main__':
    img_path = "/home/shicaiwei/data/liveness_data/cross_replayed/cross_test/spoofing/ipad/1.jpg"
    img = cv2.imread(img_path)
    # fre_phase_analysis(img, display=True)
    componenet_analysis(img=img, display=True)
