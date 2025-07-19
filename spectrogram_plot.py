import os
from scipy.signal import spectrogram, hamming
import random
import numpy as np
import matplotlib.pyplot as plt
from tftb.processing import WignerVilleDistribution
from matplotlib.colors import Normalize
from scipy.ndimage import zoom
# def plot_time_frequency(save_dir, modulations, snr):
#     """
#     从指定SNR下的每种调制方式中随机选择一条数据，并绘制其时频图
#     :param save_dir: 保存目录
#     :param modulations: 调制方式列表
#     :param snr: 指定的信噪比
#     """
#     plt.figure(figsize=(12, 12))
#     snr_str = str(int(snr))
#
#     for i, modulation in enumerate(modulations):
#         file_path = os.path.join(save_dir, modulation, snr_str)
#         if not os.path.exists(file_path):
#             print(f"未找到调制方式 {modulation} 在 SNR {snr_str}dB 下的文件夹。")
#             continue
#
#         files = os.listdir(file_path)
#         if not files:
#             print(f"调制方式 {modulation} 在 SNR {snr_str}dB 下没有数据文件。")
#             continue
#
#         file = random.choice(files)
#         iq_data = np.load(os.path.join(file_path, file))
#
#         sig_complex = iq_data[:, 0] + 1j * iq_data[:, 1]
#
#         # 计算时频图
#         f, t, Sxx = spectrogram(sig_complex, nperseg=64)
#
#         plt.subplot(6, 4, i + 1)  # 6 rows, 4 columns
#         plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
#         plt.title(f'{modulation} @ {snr_str}dB')
#         plt.xlabel('Time [sec]')
#         plt.ylabel('Frequency [Hz]')
#         plt.colorbar(label='Intensity [dB]')
#
#     plt.tight_layout()
#     plt.savefig(f"time_frequency_plots_{snr_str}dB.png")
#     plt.show()

# def plot_time_frequency(save_dir, modulations, snr):
#     plt.figure(figsize=(12, 12))
#     snr_str = str(int(snr))
#
#     for i, modulation in enumerate(modulations):
#         file_path = os.path.join(save_dir, modulation, snr_str)
#         if not os.path.exists(file_path):
#             print(f"未找到调制方式 {modulation} 在 SNR {snr_str}dB 下的文件夹。")
#             continue
#
#         files = os.listdir(file_path)
#         if not files:
#             print(f"调制方式 {modulation} 在 SNR {snr_str}dB 下没有数据文件。")
#             continue
#
#         file = random.choice(files)
#         iq_data = np.load(os.path.join(file_path, file))
#
#         # 计算时频图
#         f, t, Sxx = spectrogram(iq_data[:, 0] + 1j * iq_data[:, 1], nperseg=128)
#
#         plt.subplot(6, 4, i + 1)  # 6 rows, 4 columns
#         plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='jet')  # 使用 jet colormap
#         plt.title(f'{modulation} @ {snr_str}dB')
#         plt.xlabel('Time [sec]')
#         plt.ylabel('Frequency [Hz]')
#         plt.colorbar(label='Intensity [dB]')
#
#     plt.tight_layout()
#     plt.savefig(f"time_frequency_plots_{snr_str}dB.png")
#     plt.show()
#
#
# def main():
#     # 数据集保存路径
#     save_dir = "./IQ_signals_npy"
#
#     modulations = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
#                    '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM',
#                    '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
#                    'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
#
#     # 绘制指定SNR下每种调制方式的随机信号时频图
#     snr_to_plot = 20  # 指定SNR
#     plot_time_frequency(save_dir, modulations, snr_to_plot)




# 保存目录

def main():
    save_dir = './IQ_signals_npy'

    # 定义调制方式列表
    modulations = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                   '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM',
                   '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
                   'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']

    # 指定的SNR
    snr_to_plot = 10
    snr_str = str(snr_to_plot)

    # 创建一个新的 figure
    plt.figure(figsize=(12, 12))

    for i, modulation in enumerate(modulations):
        file_path = os.path.join(save_dir, modulation, snr_str)

        if not os.path.isdir(file_path):
            print(f'未找到调制方式 {modulation} 在 SNR {snr_str}dB 下的文件夹。')
            continue

        # 列出文件
        files = [f for f in os.listdir(file_path) if f.endswith('.npy')]
        if not files:
            print(f'调制方式 {modulation} 在 SNR {snr_str}dB 下没有数据文件。')
            continue

        # 随机选择一个文件
        chosen_file = os.path.join(file_path, random.choice(files))

        # 读取 .npy 文件
        iq_data = np.load(chosen_file)

        # 提取 I 和 Q 信号
        I_signal = iq_data[:, 0]
        Q_signal = iq_data[:, 1]
        # 将 I 和 Q 分量合成为复数信号
        # complex_signal = I_signal + 1j * Q_signal

        # 计算时频图
        # plt.subplot(6, 4, i + 1)
        # fs = 50000
        # h = hamming(128)
        # f, t, Sxx = spectrogram(complex_signal, fs=fs, window=h, nperseg=128, noverlap=127)
        # Sxx = np.fft.fftshift(Sxx, axes=0)  # 对频率轴进行FFT shift
        # f = np.linspace(-fs / 2, fs / 2, Sxx.shape[0])
        #
        # plt.pcolormesh(t, f, np.abs(Sxx), shading='gouraud')
        # plt.title(f'{modulation} @ {snr_str}dB')
        # plt.xlabel('Time [sec]')
        # plt.ylabel('Frequency [Hz]')
        # plt.colorbar()
        # wvd = WignerVilleDistribution(complex_signal)
        # tfr, t, f = wvd.run()  # 直接获取 WVD 数据和时间、频率轴
        # # 获取处理后的数据
        # # `tfr` 是复数信号处理后的时频分布矩阵
        # wvd_data = tfr
        # data_shape = wvd_data.shape  # 获取数据的形状
        # # 绘制 WVD 时频图
        # plt.pcolormesh(t, f, np.abs(tfr), shading='gouraud')
        # plt.title(f'{modulation} @ {snr_str}dB')
        # plt.xlabel('Time [sec]')
        # plt.ylabel('Frequency [Hz]')
        # plt.colorbar()


        # 使用Wigner-Ville分布绘制时频图
        plt.subplot(6, 4, i + 1)

        # 星座图分辨率设置
        image_size = 56  # 图片的像素大小
        decay_rate = 5.0  # 衰减率lambda
        # 创建二维平面并初始化像素值
        image = np.zeros((image_size, image_size, 3))  # RGB三通道
        # 归一化信号范围到图像坐标
        x_vals = np.interp(I_signal, (I_signal.min(), I_signal.max()), (0, image_size - 1)).astype(int)
        y_vals = np.interp(Q_signal, (Q_signal.min(), Q_signal.max()), (0, image_size - 1)).astype(int)
        # 计算距离衰减模型
        for i in range(1024):
            x, y = x_vals[i], y_vals[i]
            for m in range(image_size):
                for n in range(image_size):
                    dist = np.sqrt((x - m) ** 2 + (y - n) ** 2)
                    influence = 1 / (1 + decay_rate * dist)
                    image[m, n] += np.array([influence, influence, influence])

        # 将图像插值到224x224分辨率
        target_resolution = (224, 224)
        zoom_factors = (target_resolution[0] / image_size, target_resolution[1] / image_size, 1)
        resized_image = zoom(image, zoom_factors, order=3)  # 使用三次插值
        # 归一化RGB图像
        # image[:, :, 0] = Normalize()(image[:, :, 0])  # 红色通道归一化
        # image[:, :, 1] = Normalize()(image[:, :, 1])  # 绿色通道归一化
        # image[:, :, 2] = Normalize()(image[:, :, 2])  # 蓝色通道归一化
        resized_image[:, :, 0] = Normalize()(resized_image[:, :, 0])  # 红色通道归一化
        resized_image[:, :, 1] = Normalize()(resized_image[:, :, 1])  # 绿色通道归一化
        resized_image[:, :, 2] = Normalize()(resized_image[:, :, 2])  # 蓝色通道归一化
        # plt.pcolormesh(image)
        plt.pcolormesh(resized_image)

        plt.title(modulation)
        plt.colorbar()

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(f'time_frequency_plots_{snr_str}dB2.png')
    plt.show()

if __name__ == '__main__':
    main()


