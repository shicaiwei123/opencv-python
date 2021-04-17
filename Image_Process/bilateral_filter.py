import numpy as np
import cupy as cp
import cv2
import datetime
import numba as nb



def bilateral_filter_with_3dgauss(img, gaussian_kernel_3d, k_size):
    r = k_size // 2
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8

    (w, h, c) = img.shape
    filtering_img = np.zeros((w, h, c))

    # 边角不处理
    filtering_img[0:r:, :] = img[0:r, :, :]
    filtering_img[w - r, :, :] = img[w - r, :, :]
    filtering_img[:, 0:r, :] = img[:, 0:r, :]
    filtering_img[:, h - r, :] = img[:, h - r, :]

    for i in range(c):
        for j in range(r, w - r):
            for k in range(r, h - r):
                # 求值域核
                local_window = np.float32(img[j - r:j + r + 1, k - r:k + r + 1])
                local_window = np.transpose(local_window, (2, 0, 1))
                local_center = np.float32(img[j, k])
                local_center = local_center.reshape((3, 1, 1))
                value_diff = local_window - local_center
                value_kernel = np.exp(-(value_diff ** 2) / (2 * sigma ** 2))

                # 双边滤波核
                bilateral_kernel = value_kernel * gaussian_kernel_3d

                # 归一化
                normal_factor = (np.sum(bilateral_kernel, (1, 2)).reshape((3, 1, 1)))
                bilateral_kernel = bilateral_kernel / normal_factor

                # 滤波
                filtering_window = bilateral_kernel * local_window
                filtering_result = np.sum(filtering_window, axis=(1, 2))
                filtering_img[j, k] = filtering_result

    filtering_img = np.uint8(filtering_img)
    return filtering_img


@nb.jit(nopython=True)
def bilateral_filter_with_gauss(img, gaussian_kernel, k_size, x, y):
    r = k_size // 2
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8

    (w, h, c) = img.shape
    filtering_img = np.zeros((w, h, c))

    # 边角不处理
    filtering_img[0:r:, :] = img[0:r, :, :]
    filtering_img[w - r, :, :] = img[w - r, :, :]
    filtering_img[:, 0:r, :] = img[:, 0:r, :]
    filtering_img[:, h - r, :] = img[:, h - r, :]

    for i in range(c):
        img_channel = img[:, :, i]
        for j in range(r, w - r):
            for k in range(r, h - r):
                # 求值域核
                local_window = np.float32(img_channel[x, y])
                local_center = np.float32(img_channel[j, k])
                value_diff = local_window - local_center
                value_kernel = np.exp(-(value_diff ** 2) / (2 * sigma ** 2))

                # 双边滤波核
                bilateral_kernel = value_kernel * gaussian_kernel

                # 归一化
                normal_factor = np.sum(bilateral_kernel)
                bilateral_kernel = bilateral_kernel / normal_factor

                # 滤波
                filtering_window = bilateral_kernel * local_window
                filtering_result = np.sum(filtering_window)
                filtering_img[j, k] = filtering_result

    filtering_img = np.uint8(filtering_img)
    return filtering_img


def bilateral_filter(img, k_size=3):
    # 高斯核
    r = k_size // 2
    x, y = np.mgrid[-r:r + 1, -r:r + 1]
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
    gaussian_kernel = np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))
    # gaussian_kernel_a = gaussian_kernel.reshape(3, 3, 1)
    gaussian_kernel_3d = np.tile(gaussian_kernel, (3, 1))
    gaussian_kernel_3d = gaussian_kernel_3d.reshape((3, k_size, k_size))

    filtering_img = bilateral_filter_with_3dgauss(img, gaussian_kernel_3d, k_size=k_size)
    return filtering_img


if __name__ == '__main__':
    img_path = "/home/shicaiwei/project/opencv-python/data/edge_preserve/1.png"
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    cv2.namedWindow("origin", 0)
    cv2.imshow("origin", img)

    time_begin = datetime.datetime.now()
    filter_img = bilateral_filter(img, k_size=5)
    time_end = datetime.datetime.now()
    print("time", (time_end - time_begin).total_seconds())

    cv2.namedWindow("bilateral", 0)
    cv2.imshow("bilateral", filter_img)
    cv2.waitKey(0)
