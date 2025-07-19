
# 保存为.npy格式
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import random

from PIL.Image import Image


def save_iq_to_npy(iq_data, label, snr, idx, save_dir):
    """
    将IQ数据保存为.npy文件
    :param iq_data: IQ数据，形状为(1024, 2)
    :param label: 信号的调制类型标签
    :param snr: 信噪比
    :param idx: 文件索引
    :param save_dir: 保存目录
    """
    snr_str = str(int(snr))  # 去除括号，转换为字符串
    file_name = f"{label}_{snr_str}dB_{idx}.npy"
    file_path = os.path.join(save_dir, label, snr_str)

    if not os.path.exists(file_path):
        os.makedirs(file_path)
        file_index = 1  # 每个新文件夹从1开始编号
    else:
        file_index = len(os.listdir(file_path)) + 1  # 当前文件夹内文件数量加1

    # 更新文件名为按顺序编号
    file_name = f"{label}_{snr_str}dB_{file_index}.npy"

    # 保存数据
    np.save(os.path.join(file_path, file_name), iq_data)


def plot_random_samples(save_dir, modulations):
    """
    从每种调制方法中随机选择一条数据并绘制其信号序列图
    :param save_dir: 保存目录
    :param modulations: 调制方式列表
    """
    plt.figure(figsize=(12, 12))

    for i, modulation in enumerate(modulations):
        snr_folders = [f for f in os.listdir(os.path.join(save_dir, modulation)) if
                       os.path.isdir(os.path.join(save_dir, modulation, f))]
        if not snr_folders:
            continue

        # snr_folder = random.choice(snr_folders)
        snr_folder = str(10)
        files = os.listdir(os.path.join(save_dir, modulation, snr_folder))
        if not files:
            continue

        file = random.choice(files)
        file_path = os.path.join(save_dir, modulation, snr_folder, file)

        iq_data = np.load(file_path)

        plt.subplot(6, 4, i + 1)  # 6 rows, 4 columns
        plt.plot(iq_data[:, 0], label='I Channel')
        plt.plot(iq_data[:, 1], label='Q Channel')
        plt.title(modulation)
        # plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.legend()

    plt.tight_layout()
    plt.show()

modulations = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                       '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM',
                       '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
                       'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']

def main():
    # 数据集路径
    file_path = "./2018.01.OSC.0001_1024x2M.h5/GOLD_XYZ_OSC.0001_1024.hdf5"
    save_dir = "../IQ_signals_npy"

    # 读取数据时使用较小的块
    chunk_size = 15000  # 每次读取15000个样本

    # 读取数据
    with h5py.File(file_path, 'r') as f:
        Y = f['Y'][:]  # 标签数据
        Z = f['Z'][:]  # SNR数据
        total_samples = f['X'].shape[0]  # 样本总数



        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            X_chunk = f['X'][start_idx:end_idx]  # 读取块数据

            for i in range(X_chunk.shape[0]):
                label_idx = np.argmax(Y[start_idx + i])  # 获取调制类型索引
                modulation = modulations[label_idx]
                snr = Z[start_idx + i]

                # 保存每个样本的IQ数据
                save_iq_to_npy(X_chunk[i], modulation, snr, i, save_dir)

    # 绘制每种调制方法下的随机信号序列图
    plot_random_samples(save_dir, modulations)


if __name__ == '__main__':
    main()
