## 测试格拉姆角场绘制
import os
# import numpy as np
# import random
# import matplotlib.pyplot as plt
# def inspect_and_plot_random_npy_file(save_path):
#     """
#     Randomly select an .npy file from the save path, output its content and shape,
#     and plot the image.
#
#     Parameters:
#         save_path (str): Path to the directory containing saved .npy files.
#     """
#
#     selected_file_path = save_path
#
#     # Load the .npy file
#     data = np.load(selected_file_path)
#
#     # Output the shape and content of the loaded .npy file
#     print(f"Selected file: {selected_file_path}")
#     print(f"Shape of the array: {data.shape}")
#
#     # Plot the image
#     plt.figure(figsize=(5, 5))
#     plt.imshow(data, origin='lower')  # Use 'origin' to correctly display the image orientation
#     plt.title(f'Gramian Angular Field (GAF) Image\n{save_path}')
#     plt.axis('off')  # Hide axes for better visualization
#     plt.show()
#
# 测试格拉姆角场绘制
# Specify the save path
# save_path = '../dataset3/train/geram/4ASK/4ASK_-2dB_3117.npy'
# save_path = '../dataset3/train/geram/8PSK/8PSK_-2dB_105.npy'
# save_path = '../dataset3/train/geram/AM-DSB-SC/AM-DSB-SC_-4dB_1351.npy'
# save_path = '../dataset3/test/geram/OOK/-2/OOK_-2dB_4076.npy'
#
# # Run the inspection and plotting
# inspect_and_plot_random_npy_file(save_path)


## 测试星座图绘制
import numpy as np
# import matplotlib.pyplot as plt
#
# # 生成示例 IQ 信号（长度为 1024）
# I = np.random.randn(1024)
# Q = np.random.randn(1024)
#
# # 绘制星座图
# plt.figure(figsize=(8, 8))
# plt.scatter(I, Q, s=10, color='blue')
# plt.title('Constellation Diagram')
# plt.xlabel('In-Phase (I)')
# plt.ylabel('Quadrature (Q)')
# plt.grid(True)
# plt.axis('equal')  # 保证 x 轴和 y 轴的比例相同
# plt.show()

# # 测试WVD时频图
# import numpy as np
# import matplotlib.pyplot as plt
# from tftb.processing import WignerVilleDistribution
#
# # 生成示例 IQ 信号
# iq_signal = np.random.randn(1024, 2)
# complex_signal = iq_signal[:, 0] + 1j * iq_signal[:, 1]
#
# # 计算 Wigner-Ville 分布
# wvd = WignerVilleDistribution(complex_signal)
# tfr, t, f = wvd.run()  # 直接获取 WVD 数据和时间、频率轴
#
# # 获取处理后的数据
# # `tfr` 是复数信号处理后的时频分布矩阵
# wvd_data = tfr
#
# # 获取数据维度和尺寸
# data_shape = wvd_data.shape  # 获取数据的形状
# num_time_points = data_shape[0]  # 时间点数
# num_frequency_points = data_shape[1]  # 频率点数
#
# # 打印结果
# print(f"处理后的数据形状: {data_shape}")
# print(f"时间点数: {num_time_points}")
# print(f"频率点数: {num_frequency_points}")
#
# # 绘制 WVD 时频图
# plt.figure(figsize=(10, 6))
# plt.imshow(np.abs(wvd_data), aspect='auto', cmap='jet', origin='lower', extent=[t.min(), t.max(), f.min(), f.max()])
# plt.colorbar(label='Magnitude')
# plt.title('Wigner-Ville Distribution')
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import zoom
# 加载保存的图像
loaded_image = np.load('../dataset/train/star/OOK/OOK_0dB_1924.npy')

# 显示插值后的图像
plt.imshow(loaded_image)
plt.title('Enhanced Constellation Diagram at 224x224 Resolution')
plt.show()
