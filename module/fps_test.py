import os
import numpy as np
import time
from scipy.signal import spectrogram, hamming
from PIL import Image
import torch
from torch.utils.data import Dataset


class OptimizedSignalDataset(Dataset):
    def __init__(self, npy_dir):
        """
        npy_dir: 信号npy文件存放路径
        """
        self.npy_dir = npy_dir
        self.data = []

        # 处理数据
        for npy_file in os.listdir(npy_dir):
            if npy_file.endswith('.npy'):
                npy_filepath = os.path.join(npy_dir, npy_file)
                self.data.append(npy_filepath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        npy_filepath = self.data[idx]
        signal = np.load(npy_filepath)  # 读取npy信号 (1024, 2)，包括I和Q信号
        return signal

    # 归一化信号函数
    def normalize_signal(self, signal):
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) * 2 - 1

    # 计算角度函数
    def convert_to_angle(self, signal):
        return np.arccos(signal)

    # 计算Gramian Angular Summation Field (GASF)
    def compute_gasf(self, I_angle, Q_angle):
        length = len(I_angle)
        gasf = np.zeros((length, length))
        for i in range(length):
            for j in range(length):
                gasf[i, j] = np.cos(I_angle[i] - I_angle[j]) + np.cos(Q_angle[i] - Q_angle[j])
        return gasf

    # 计算Gramian Angular Difference Field (GADF)
    def compute_gadf(self, I_angle, Q_angle):
        length = len(I_angle)
        gadf = np.zeros((length, length))
        for i in range(length):
            for j in range(length):
                gadf[i, j] = np.sin(I_angle[i] - I_angle[j]) + np.sin(Q_angle[i] - Q_angle[j])
        return gadf

    def generate_spectrogram(self, signal):
        """Generate spectrogram from I/Q signal and return as numpy array"""
        # 提取 I 和 Q 信号
        I_signal = signal[:, 0]
        Q_signal = signal[:, 1]

        # 确保窗口大小小于信号长度
        window_size = 32
        overlap = window_size // 2  # 通常使用窗口大小的一半作为重叠

        # 计算时频图
        fs = 256  # 采样率，根据实际数据调整
        h = hamming(window_size)
        f, t, Sxx = spectrogram(I_signal + 1j * Q_signal, fs=fs, window=h, nperseg=window_size, noverlap=overlap)
        Sxx = np.fft.fftshift(Sxx, axes=0)  # 对频率轴进行FFT shift

        # 对时频图进行归一化处理，确保在合理的数值范围
        Sxx = 10 * np.log10(Sxx + 1e-8)  # 转换为dB，避免log(0)的情况

        # 归一化到0-255范围
        Sxx_min, Sxx_max = Sxx.min(), Sxx.max()
        Sxx_normalized = (Sxx - Sxx_min) / (Sxx_max - Sxx_min)  # 归一化到0-1
        Sxx_normalized = (Sxx_normalized * 255).astype(np.uint8)  # 再转为0-255范围的uint8

        # 转换为3通道图像
        Sxx_3ch = np.stack([Sxx_normalized] * 3, axis=-1)

        # 确保尺寸为224x224
        if Sxx_3ch.shape[0] != 224 or Sxx_3ch.shape[1] != 224:
            data = np.array(Image.fromarray(Sxx_3ch).resize((224, 224)))
        else:
            data = Sxx_3ch

        return data

    def generate_gaf(self, signal):
        """Generate GAF from I/Q signal and return as numpy array"""
        # 提取 I 和 Q 信号
        I_signal = signal[:, 0]
        Q_signal = signal[:, 1]

        # 归一化信号
        I_normalized = self.normalize_signal(I_signal)
        Q_normalized = self.normalize_signal(Q_signal)

        # 转换为角度
        I_angle = self.convert_to_angle(I_normalized)
        Q_angle = self.convert_to_angle(Q_normalized)

        # 计算GASF和GADF
        gasf_image = self.compute_gasf(I_angle, Q_angle).astype(np.uint8)
        # gadf_image = self.compute_gadf(I_angle, Q_angle).astype(np.uint8)

        gaf_3ch = np.stack([gasf_image] * 3, axis=-1)

        # 确保尺寸为 224x224
        if gaf_3ch.shape[0] != 224 or gaf_3ch.shape[1] != 224:
            data = np.array(Image.fromarray(gaf_3ch).resize((224, 224)))
        else:
            data = gaf_3ch

        return data

# 配置路径
train_dir = '../dataset/train/OOK'
dataset = OptimizedSignalDataset(train_dir)

# 计算时间
num_samples = 1
start_time = time.time()

for i in range(num_samples):
    signal = dataset[i]
    _ = dataset.generate_spectrogram(signal)

spectrogram_time = time.time() - start_time

start_time = time.time()

for i in range(num_samples):
    signal = dataset[i]
    _ = dataset.generate_gaf(signal)

gaf_time = time.time() - start_time

print(f"Time for computing spectrogram: {spectrogram_time:.2f} seconds")
print(f"Time for computing GAF: {gaf_time:.2f} seconds")