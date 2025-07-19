import os
import numpy as np
import torch
from PIL import Image
from pyts.approximation import PiecewiseAggregateApproximation
from pyts.preprocessing import MinMaxScaler
from matplotlib.colors import Normalize
from scipy.ndimage import zoom

modulation_types = [
    'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
    '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM',
    '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
    'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK'
]

def calculate_star(I_signal, Q_signal):
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
    return resized_image

def process_train_dataset(trainset_path, train_save_path):
    for modulation in modulation_types:
        # Directory for the current modulation type
        modulation_dir = os.path.join(trainset_path, modulation)
        save_modulation_dir = os.path.join(train_save_path, modulation)
        os.makedirs(save_modulation_dir, exist_ok=True)

        # List all .npy files in the directory
        npy_files = [f for f in os.listdir(modulation_dir) if f.endswith('.npy')]

        for npy_file in npy_files:
            # Load the .npy file
            iq_data = np.load(os.path.join(modulation_dir, npy_file))  # Shape should be 2x1024
            iq_data = torch.tensor(iq_data.T, dtype=torch.float32)
            # Extract I and Q signals
            I_signal = iq_data[0, :].numpy()
            Q_signal = iq_data[1, :].numpy()

            # Combine I and Q GAFs into an RGB image
            star_gaf = calculate_star(I_signal, Q_signal)

            # Save the combined GAF as a .npy file
            save_file_path = os.path.join(save_modulation_dir, npy_file)
            np.save(save_file_path, star_gaf)

    pass

def process_test_dataset(testset_path, test_save_path):
    """
    处理测试集中的信号，计算它们的 GAF，并保存到指定路径。

    参数:
        testset_path (str): 测试集路径，其中包含按调制方式和 SNR 分类的信号。
        test_save_path (str): GAF 计算结果保存路径。
    """
    # # 遍历调制方式文件夹
    # modulation_types = os.listdir(testset_path)
    for modulation in modulation_types:
        modulation_dir = os.path.join(testset_path, modulation)
        if not os.path.isdir(modulation_dir):
            continue

        # 遍历 SNR 文件夹
        snr_values = os.listdir(modulation_dir)
        for snr in snr_values:
            snr_dir = os.path.join(modulation_dir, snr)
            if not os.path.isdir(snr_dir):
                continue

            # 遍历 SNR 文件夹中的 .npy 文件
            npy_files = [f for f in os.listdir(snr_dir) if f.endswith('.npy')]
            for npy_file in npy_files:
                # 加载 .npy 文件
                iq_data_path = os.path.join(snr_dir, npy_file)
                iq_data = np.load(iq_data_path)  # Shape should be 2x1024
                iq_data = torch.tensor(iq_data.T, dtype=torch.float32)

                # 提取 I 和 Q 信号
                I_signal = iq_data[0, :].numpy()
                Q_signal = iq_data[1, :].numpy()

                # 计算 I 和 Q 的 GAF，并创建 RGB 图像
                star_rgb = calculate_star(I_signal, Q_signal)

                # 构建保存路径
                save_modulation_dir = os.path.join(test_save_path, modulation)
                save_snr_dir = os.path.join(save_modulation_dir, snr)
                os.makedirs(save_snr_dir, exist_ok=True)

                # 保存为 .npy 文件
                save_path = os.path.join(save_snr_dir, npy_file)
                np.save(save_path, star_rgb)


def main():
    # Paths
    trainset_path = '../dataset/train/signal'
    testset_path = '../dataset/test/signal'
    train_save_path = '../dataset/train/star'
    test_save_path = '../dataset/test/star'

    # List of modulation types (folders)
    process_train_dataset(trainset_path, train_save_path)
    process_test_dataset(testset_path, test_save_path)

if __name__ == "__main__":
    main()


