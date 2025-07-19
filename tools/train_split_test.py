# 将train中的文件拷贝为test文件格式，用做测试，检测为什么train和test准确率相差很多

import os
import shutil
import re

# Paths to the original 'geram' and 'signal' folders
base_train_path = '../dataset/train'
geram_path = os.path.join(base_train_path, 'geram')
signal_path = os.path.join(base_train_path, 'signal')

# Paths to the destination 'geram2' and 'signal2' folders
geram2_path = os.path.join(base_train_path, 'geram2')
signal2_path = os.path.join(base_train_path, 'signal2')

# Ensure the new directories exist
os.makedirs(geram2_path, exist_ok=True)
os.makedirs(signal2_path, exist_ok=True)


# Modified function to copy modulation folders and save files according to SNR values
# def copy_modulation_folders_by_snr(src_folder, dest_folder):
#     for modulation_type in os.listdir(src_folder):
#         modulation_folder = os.path.join(src_folder, modulation_type)
#         if os.path.isdir(modulation_folder):
#             # Process .npy files to extract SNR values and copy them
#             for file in os.listdir(modulation_folder):
#                 if file.endswith('.npy'):
#                     # Extract the SNR value from the file name
#                     # match = re.search(r'_snr_(-?\d+)\.npy$', file)
#                     match = re.search(r'_(\d+)dB_', file)
#                     if match:
#                         snr = str(match.group(1))
#                         # Create destination folder based on SNR value
#                         dest_modulation_folder = os.path.join(dest_folder, modulation_type, snr)
#                         os.makedirs(dest_modulation_folder, exist_ok=True)
#                         # Copy file to the destination folder with the same name
#                         src_file_path = os.path.join(modulation_folder, file)
#                         dest_file_path = os.path.join(dest_modulation_folder, file)
#                         shutil.copy(src_file_path, dest_file_path)

# Copy data from 'geram' to 'geram2' using the modified function

import os
import shutil

def copy_modulation_folders_by_snr(src_folder, dest_folder):
    for modulation_type in os.listdir(src_folder):
        modulation_folder = os.path.join(src_folder, modulation_type)
        if os.path.isdir(modulation_folder):
            # Process .npy files to extract SNR values and copy them
            for file in os.listdir(modulation_folder):
                if file.endswith('.npy'):
                    # 使用'_'划分文件名并提取SNR值
                    parts = file.split('_')
                    try:
                        # 找到包含'dB'的部分，然后取出SNR值
                        for part in parts:
                            if 'dB' in part:
                                snr = int(part.replace('dB', ''))
                                break
                        else:
                            # 如果没有找到包含'dB'的部分，跳过该文件
                            continue

                        # Create destination folder based on SNR value
                        dest_modulation_folder = os.path.join(dest_folder, modulation_type, str(snr))
                        os.makedirs(dest_modulation_folder, exist_ok=True)
                        # Copy file to the destination folder with the same name
                        src_file_path = os.path.join(modulation_folder, file)
                        dest_file_path = os.path.join(dest_modulation_folder, file)
                        shutil.copy(src_file_path, dest_file_path)
                    except ValueError:
                        # 如果转换失败，跳过该文件
                        continue

copy_modulation_folders_by_snr(geram_path, geram2_path)

# Copy data from 'signal' to 'signal2' using the modified function
copy_modulation_folders_by_snr(signal_path, signal2_path)

