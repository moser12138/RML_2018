# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import spectrogram, hamming
#
#
# def process_modulation(modulation, save_dir, spec_dir):
#     """
#     对每种调制方式下的指定SNR范围内的所有IQ信号绘制时频图并保存
#     :param modulation: 调制方式
#     :param save_dir: IQ信号数据的保存目录
#     :param spec_dir: 时频图保存目录
#     """
#     modulation_dir = os.path.join(save_dir, modulation)
#
#     # 定义要处理的SNR值范围
#     snr_values = list(range(-10, 21, 2))
#
#     # 创建调制方式的保存目录
#     modulation_spec_dir = os.path.join(spec_dir, modulation)
#     if not os.path.isdir(modulation_spec_dir):
#         os.makedirs(modulation_spec_dir)
#
#     # 遍历每个SNR文件夹
#     img_counter = 1
#     for snr_value in snr_values:
#         snr_folder = str(snr_value)
#         file_path = os.path.join(modulation_dir, snr_folder)
#
#         if not os.path.isdir(file_path):
#             print(f'调制方式 {modulation} 在 SNR {snr_value}dB 下没有数据文件夹。')
#             continue
#
#         # 列出所有 .npy 文件
#         files = [f for f in os.listdir(file_path) if f.endswith('.npy')]
#         if not files:
#             print(f'调制方式 {modulation} 在 SNR {snr_value}dB 下没有数据文件。')
#             continue
#
#         # 遍历每个文件
#         for file_name in sorted(files):
#             chosen_file = os.path.join(file_path, file_name)
#
#             # 读取 .npy 文件
#             iq_data = np.load(chosen_file)
#
#             # 提取 I 和 Q 信号
#             I_signal = iq_data[:, 0]
#             Q_signal = iq_data[:, 1]
#
#             # 计算时频图
#             fs = 256
#             h = hamming(128)
#             f, t, Sxx = spectrogram(I_signal + 1j * Q_signal, fs=fs, window=h, nperseg=128, noverlap=127)
#             Sxx = np.fft.fftshift(Sxx, axes=0)  # 对频率轴进行FFT shift
#             f = np.linspace(-fs / 2, fs / 2, Sxx.shape[0])
#
#             # 创建图像
#             fig, ax = plt.subplots(figsize=(224 / 100, 224 / 100), dpi=100)
#             img = ax.imshow(np.abs(Sxx), aspect='auto', origin='lower', cmap='inferno',
#                             extent=[t.min(), t.max(), f.min(), f.max()])
#             ax.axis('off')  # 不显示坐标轴
#
#             # 保存图像
#             image_filename = os.path.join(modulation_spec_dir, f'{modulation}_{img_counter}.png')
#             plt.savefig(image_filename, bbox_inches='tight', pad_inches=0, dpi=100)
#             plt.close(fig)  # 关闭当前图像以释放内存
#
#             img_counter += 1  # 更新编号
#
#
# def main():
#     save_dir = 'IQ_signals_npy'
#     spec_dir = 'spec'
#
#     # 创建图像保存目录
#     if not os.path.isdir(spec_dir):
#         os.makedirs(spec_dir)
#     # modulations = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
#         #                '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM',
#         #                '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
#         #                'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
#     modulations = ['4ASK', 'BPSK', 'QPSK', '16QAM', '32QAM', 'AM-SSB-SC', 'AM-DSB-SC']
#
#     # 对每种调制方式处理
#     for modulation in modulations:
#         process_modulation(modulation, save_dir, spec_dir)
#
#
# if __name__ == '__main__':
#     main()
#
#

import os
import numpy as np
from scipy.signal import spectrogram, hamming
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

def process_modulation(modulation, save_dir, final_dir):
    """
    对每种调制方式下的指定SNR范围内的所有IQ信号绘制时频图并调整为指定尺寸
    :param modulation: 调制方式
    :param save_dir: IQ信号数据的保存目录
    :param final_dir: 最终图像文件夹
    """
    modulation_dir = os.path.join(save_dir, modulation)

    # 定义要处理的SNR值范围
    snr_values = list(range(-8, 16, 2))

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

        # 只读取前1500条信号
        files = sorted(files)[:1500]

        # 遍历每个文件
        for file_name in files:
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

            # 使用 BytesIO 保存图像到内存
            with BytesIO() as img_bytes:
                fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
                ax.imshow(np.abs(Sxx), aspect='auto', origin='lower', cmap='inferno',
                          extent=[t.min(), t.max(), f.min(), f.max()])
                ax.axis('off')  # 不显示坐标轴
                plt.savefig(img_bytes, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
                plt.close(fig)  # 关闭当前图像以释放内存

                img_bytes.seek(0)  # 重置 BytesIO 的位置到开头

                # 打开图像并调整尺寸
                with Image.open(img_bytes) as img:
                    img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
                    final_image_path = os.path.join(modulation_final_dir, f'{modulation}_{img_counter}.png')
                    img_resized.save(final_image_path, format='PNG')

            img_counter += 1  # 更新编号

def main():
    save_dir = 'IQ_signals_npy'
    final_dir = 'spec2'

    # 创建临时和最终图像文件夹
    os.makedirs(final_dir, exist_ok=True)

    # 定义调制方式列表
    modulations = ['4ASK', 'BPSK', 'QPSK', '16QAM', '32QAM', 'AM-SSB-SC', 'AM-DSB-SC']

    # 对每种调制方式处理
    for modulation in modulations:
        process_modulation(modulation, save_dir, final_dir)


if __name__ == '__main__':
    main()




