import os
import shutil
import random
from pathlib import Path

# 原始数据集路径
source_dir = '../IQ_signals_npy'

# 目标train和test目录
train_dir = '../dataset/train'
test_dir = '../dataset/test'

# 需要选定的调制方式文件夹
modulations = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
               '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM',
               '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
               'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']

# 选定的SNR范围及间隔
snr_range = list(range(-20, 12, 2))  # 包含从-20到10，步长为2

# 创建目标目录，如果不存在则创建
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 设置随机种子，以确保每次分割的结果一致
random.seed(42)


def split_files_in_folder(src_folder, train_folder, test_folder):
    """
    将src_folder中的npy文件随机选择1000个放入train_folder，250个放入test_folder
    """
    # 获取当前SNR文件夹中的所有npy文件
    npy_files = [f for f in os.listdir(src_folder) if f.endswith('.npy')]

    # 如果npy文件少于1000 + 250个，直接返回
    if len(npy_files) < 1250:
        return

    # 随机打乱并选择1000个用于训练，250个用于测试
    random.shuffle(npy_files)
    train_files = npy_files[:1000]
    test_files = npy_files[1000:1250]

    # 将train文件复制到调制方式文件夹，不按SNR分类
    for npy_file in train_files:
        src_file = os.path.join(src_folder, npy_file)
        dst_file = os.path.join(train_folder, npy_file)  # 不再区分SNR，直接存储
        shutil.copy2(src_file, dst_file)

    # 将test文件复制到目标文件夹，仍按SNR分类
    for npy_file in test_files:
        src_file = os.path.join(src_folder, npy_file)
        dst_file = os.path.join(test_folder, npy_file)  # 仍按SNR存储
        shutil.copy2(src_file, dst_file)


def process_modulation_folder(modulation_folder):
    """
    处理每个调制方式的文件夹，划分其SNR文件夹下的npy文件
    """
    # 为当前调制方式创建train文件夹（不再按SNR分类）
    train_mod_path = modulation_folder.replace(source_dir, train_dir)
    os.makedirs(train_mod_path, exist_ok=True)

    # 遍历每个调制方式文件夹下的SNR文件夹
    for snr_folder in os.listdir(modulation_folder):
        try:
            snr_value = int(snr_folder)
        except ValueError:
            continue  # 如果无法转换为整数，跳过此SNR文件夹

        # 只处理SNR在指定范围内的文件夹
        if snr_value in snr_range:
            src_snr_path = os.path.join(modulation_folder, snr_folder)

            if os.path.isdir(src_snr_path):
                # 为当前SNR文件夹创建对应的test文件夹（按SNR分类）
                test_snr_path = os.path.join(test_dir, Path(modulation_folder).name, snr_folder)
                os.makedirs(test_snr_path, exist_ok=True)

                # 分割当前SNR文件夹中的npy文件到train文件夹和test文件夹
                split_files_in_folder(src_snr_path, train_mod_path, test_snr_path)


# 主函数
def main():
    # 遍历选定的调制方式
    for mod in modulations:
        mod_path = os.path.join(source_dir, mod)

        if os.path.isdir(mod_path):
            print(f"Processing modulation type: {mod}")
            process_modulation_folder(mod_path)


if __name__ == '__main__':
    main()
