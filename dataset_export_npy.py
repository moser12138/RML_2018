import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, hamming
from PIL import Image, ImageResampling


def process_modulation(modulation, save_dir, final_dir):
    """
    对每种调制方式下的指定SNR范围内的所有IQ信号绘制时频图，调整尺寸并直接保存最终图片
    :param modulation: 调制方式
    :param save_dir: IQ信号数据的保存目录
    :param final_dir: 最终图像文件夹
    """
    modulation_dir = os.path.join(save_dir, modulation)

    # 定义要处理的SNR值范围
    snr_values = list(range(-10, 21, 2))

    # 创建调制方式的最终保存目录
    modulation_final_dir = os.path.join(final_dir, modulation)
    os.makedirs(modulation_final_dir, exist_ok=True)

    img_counter = 1
    for snr_value in snr_values:
        snr_folder = str(snr_value)
        file_path = os.path.join(modulation_dir, snr_folder)

        if not os.path.isdir(file_path):
            print(f'调制方式 {modulation} 在 SNR {snr_value}dB 下没有数据文件夹。')
            continue

        # 列出所有 .npy 文件
        files = [f for f in os.listdir(file_path) if f.endswith('.npy')]
        if not files:
            print(f'调制方式 {modulation} 在 SNR {snr_value}dB 下没有数据文件。')
            continue

        # 遍历每个文件
        for file_name in sorted(files):
            chosen_file = os.path.join(file_path, file_name)

            # 读取 .npy 文件
            iq_data = np.load(chosen_file)

            # 提取 I 和 Q 信号
            I_signal = iq_data[:, 0]
            Q_signal = iq_data[:, 1]

            # 计算时频图
            fs = 256
            h = hamming(128)
            f, t, Sxx = spectrogram(I_signal + 1j * Q_signal, fs=fs, window=h, nperseg=128, noverlap=127)
            Sxx = np.fft.fftshift(Sxx, axes=0)  # 对频率轴进行FFT shift
            f = np.linspace(-fs / 2, fs / 2, Sxx.shape[0])

            # 创建图像
            fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
            ax.imshow(np.abs(Sxx), aspect='auto', origin='lower', cmap='inferno',
                      extent=[t.min(), t.max(), f.min(), f.max()])
            ax.axis('off')  # 不显示坐标轴

            # 保存图像并调整尺寸
            temp_image_filename = os.path.join(modulation_final_dir, f'{modulation}_{img_counter}.png')
            plt.savefig(temp_image_filename, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close(fig)  # 关闭当前图像以释放内存

            # 调整图片尺寸
            with Image.open(temp_image_filename) as img:
                img_resized = img.resize((224, 224), resample=Image.Resampling.LANCZOS)
                img_resized.save(temp_image_filename, format='PNG')

            img_counter += 1  # 更新编号


def main():
    save_dir = 'IQ_signals_npy'
    final_dir = 'spec2'

    # 创建最终图像文件夹
    os.makedirs(final_dir, exist_ok=True)

    # 定义调制方式列表
    modulations = ['4ASK', 'BPSK', 'QPSK', '16QAM', '32QAM', 'AM-SSB-SC', 'AM-DSB-SC']

    # 对每种调制方式处理
    for modulation in modulations:
        process_modulation(modulation, save_dir, final_dir)


if __name__ == '__main__':
    main()
