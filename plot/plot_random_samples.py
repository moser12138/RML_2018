import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import random
from PIL.Image import Image

# 设置字体路径（根据你系统中文和英文字体的路径来）
times_path = "/home/ll/.fonts/TimesNewRoman/Times New Roman.ttf"  # 或 Windows: "C:/Windows/Fonts/times.ttf"

times_font = fm.FontProperties(fname=times_path)  # 英文 Times New Roman

# 设置 matplotlib 全局字体
plt.rcParams['font.family'] = times_font.get_name()


def plot_random_samples(save_dir, modulations):
    """
    从每种调制方法中随机选择一条数据并绘制其信号序列图
    :param save_dir: 保存目录
    :param modulations: 调制方式列表
    """
    plt.figure(figsize=(15, 16))

    for i, modulation in enumerate(modulations):
        snr_folders = [f for f in os.listdir(os.path.join(save_dir, modulation)) if
                       os.path.isdir(os.path.join(save_dir, modulation, f))]
        if not snr_folders:
            continue

        # snr_folder = random.choice(snr_folders)
        snr_folder = str(30)
        files = os.listdir(os.path.join(save_dir, modulation, snr_folder))
        if not files:
            continue

        file = random.choice(files)
        file_path = os.path.join(save_dir, modulation, snr_folder, file)

        iq_data = np.load(file_path)
        plt.tick_params(axis='both', labelsize=16)  # 设置横纵坐标数字的字体大小为12

        plt.subplot(6, 4, i + 1)  # 6 rows, 4 columns
        plt.plot(iq_data[:, 0], label='I Channel')
        plt.plot(iq_data[:, 1], label='Q Channel')
        plt.title(modulation, fontproperties=times_font, fontsize=25)

        # 修改图例、坐标轴刻度等使用 Times New Roman 字体
        plt.legend(loc='lower right', prop=times_font, fontsize=25)  # 图例字体
        # plt.xlabel('Sample Index', fontproperties=times_font, fontsize=14)  # x轴标签
        # plt.ylabel('Amplitude', fontproperties=times_font, fontsize=14)  # y轴标签
        plt.xticks(fontproperties=times_font, fontsize=16)  # x轴刻度
        plt.yticks(fontproperties=times_font, fontsize=16)  # y轴刻度

    # 保存图像到文件
    plt.tight_layout()
    plt.savefig('data_npy', dpi=600)
    plt.show()


modulations = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
               '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM',
               '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
               'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']


def main():
    # 数据路径
    save_dir = "../IQ_signals_npy"

    # 绘制每种调制方法下的随机信号序列图
    plot_random_samples(save_dir, modulations)


if __name__ == '__main__':
    main()
