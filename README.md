# opencv-python
长期更新，利用opencv的python版本去进行各种传统图像处理操作的代码，以文件夹的形式区分保存

# devlopemt

# 框架介绍
- 目前入口文件为 mutil_img_enchace.py
- data
    - 测试中用用到的各种数据
- 支持
    - Image_Process/retinax.py: retianx图像增强, SSR MSR
    - Image_Process/histogramEqualization.py: 直方图增强,自动直方图均衡,全局直方图均衡,CLAHE,gamma校正
    - Image_Process/edge_preserve.py: 双边滤波和引导滤波(简单版本和加速版本)
    - Image_Quality/mutil_img_enchace.py: 多帧等比合并,综合所有增强的处理
    - Image_Quality/edge_enhance.py: 边缘增强, usm
    - Frequency_Analysis/frequency_component.py : 绘制频谱，相位图，显示图像低通高通相位频谱图
    -