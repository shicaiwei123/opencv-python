# opencv-python
长期更新，利用opencv的python版本去进行各种传统图像处理操作的代码，以文件夹的形式区分保存

# devlopemt

# 框架介绍
- 目前入口文件为 mutil_img_enchace.py
- data
    - 测试中用用到的各种数据
- 支持
    - retinax.py: retianx图像增强, SSR MSR
    - histogramEqualization.py: 直方图增强:,自动直方图均衡,全局直方图均衡,CLAHE,gamma校正
    - mutil_img_enchace.py: 多帧等比合并,综合所有增强的处理
    - edge_enhance.py: 边缘增强, usm
    - Frequency_Analysis/frequency_component.py : 绘制频谱，相位图，显示图像低通高通相位频谱图