import os
import numpy as np
import torch
from PIL import Image
from pyts.approximation import PiecewiseAggregateApproximation
from pyts.preprocessing import MinMaxScaler

modulation_types = [
    'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
    '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM',
    '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
    'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK'
]

def compute_gaf(signal):
    """
    计算单个信号的格拉姆角场 (GAF)。

    参数:
        signal (numpy.ndarray): 信号数据的一维数组。

    返回:
        numpy.ndarray: 格拉姆角场的二维数组。
    """
    # 确保信号是 NumPy 数组
    if isinstance(signal, torch.Tensor):
        signal = signal.numpy()

    # 分段聚合近似 (PAA)
    transformer = PiecewiseAggregateApproximation(window_size=2)
    result = transformer.transform([signal])

    # 将数据缩放到区间 [0, 1]
    scaler = MinMaxScaler()
    scaled_X = scaler.transform(result)

    # 对缩放后的数据进行裁剪，以确保在 [0, 1] 范围内
    scaled_X = np.clip(scaled_X, 0, 1)

    # 转换为格拉姆角场
    # 在计算 arccos 之前对数据进行裁剪，以确保在 [-1, 1] 范围内
    arccos_X = np.arccos(np.clip(scaled_X[0, :], -1, 1))

    field = [a + b for a in arccos_X for b in arccos_X]
    gram = np.cos(field).reshape(len(arccos_X), len(arccos_X))

    return gram

def create_rgb_gaf_image(I_signal, Q_signal):
    """
    Create an RGB image where the I and Q signals' GAFs are the R and G channels,
    and the B channel is filled with zeros.

    Parameters:
        I_signal (numpy.ndarray): 1D array of the I signal data.
        Q_signal (numpy.ndarray): 1D array of the Q signal data.

    Returns:
        numpy.ndarray: A 3D array representing the RGB image.
    """
    # Compute GAFs for I and Q signals
    I_gaf = compute_gaf(I_signal)
    Q_gaf = compute_gaf(Q_signal)

    # Create an empty channel with the same shape as the GAFs
    empty_channel = np.zeros_like(I_gaf)

    # Combine I_gaf, Q_gaf, and empty_channel into an RGB image
    rgb_image = np.stack([I_gaf, Q_gaf, empty_channel], axis=-1)

    rgb_image = np.clip(rgb_image, 0, 1)  # Ensure the data is in the [0, 1] range

    gaf_rgb_uint8 = (rgb_image * 255).astype(np.uint8)  # 转换为 0-255 范围的 uint8

    # 确保尺寸为 224x224
    if gaf_rgb_uint8.shape[0] != 224 or gaf_rgb_uint8.shape[1] != 224:
        data = np.array(Image.fromarray(gaf_rgb_uint8).resize((224, 224)))
    else:
        data = gaf_rgb_uint8

    return data


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
            combined_gaf = create_rgb_gaf_image(I_signal, Q_signal)

            # Save the combined GAF as a .npy file
            save_file_path = os.path.join(save_modulation_dir, npy_file)
            np.save(save_file_path, combined_gaf)

    pass

def process_test_dataset(testset_path, test_save_path):
    """
    处理测试集中的信号，计算它们的 GAF，并保存到指定路径。

    参数:
        testset_path (str): 测试集路径，其中包含按调制方式和 SNR 分类的信号。
        test_save_path (str): GAF 计算结果保存路径。
    """
    # 遍历调制方式文件夹
    modulation_types = os.listdir(testset_path)
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
                gaf_rgb = create_rgb_gaf_image(I_signal, Q_signal)

                # 构建保存路径
                save_modulation_dir = os.path.join(test_save_path, modulation)
                save_snr_dir = os.path.join(save_modulation_dir, snr)
                os.makedirs(save_snr_dir, exist_ok=True)

                # 保存为 .npy 文件
                save_path = os.path.join(save_snr_dir, npy_file)
                np.save(save_path, gaf_rgb)


def main():
    # Paths
    trainset_path = '../dataset/train/signal'
    testset_path = '../dataset/test/signal'
    train_save_path = '../dataset/train/geram'
    test_save_path = '../dataset/test/geram'

    # List of modulation types (folders)
    # process_train_dataset(trainset_path, train_save_path)
    process_test_dataset(testset_path, test_save_path)

if __name__ == "__main__":
    main()


