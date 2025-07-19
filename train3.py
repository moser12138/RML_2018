 # 在使用代码将RML_2018数据集导出为.npy文件后，到处信号存储在IQ_signals_npy文件夹目录下，目录中按照信号调制类型分为不同文件夹，在不同调制类型文件夹中按照snr数值分为不同文件夹，请帮我写一个完整的python信号分类代码，要求不使用官方模型，自己编写一个完整的双模态分类模型，一种输入类型为信号的npy文件，另一个输入为通过信号npy文件画出的信号时频图，时频图尺寸规定为224*224，两路输入后，进入蛇毒学习网络训练，并最后特征融合，共同决定分类标签，请打包函数，写出main函数，分别写出读取数据集，训练，测试，等功能，完整的进行训练
# 请改进其中的train与test函数，要求训练过程中每10轮会验证一次当前模型参数的分类准确率，并与已经保存的参数准确率比较，如果当前准确率较高，则会替换保存的参数，并且训练和测试过程中会用tqdm中进度条方式展示训练进度，每结束一轮，会在log文件中保存当前轮训练结果，10轮验证时也会保存结果，训练中需要展示loss，准确率，一轮花费时间等参数
# 请基于以下代码修改，将绘制时频图代码写在神经网络中，输入网络的为两路IQ信号，先将IQ信号转化为时频图，输入图像模态网络分支，再将IQ信号序列输入序列网络分支进行综合分类操纵，IQ信号尺度分为两路，每路长度1024,请将其转为224*224的时频图
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, hamming
from PIL import Image

import numpy as np
from scipy.signal import spectrogram
from PIL import Image
import io
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm  # 用于显示进度条
import os
from torch.utils.data import Dataset
import numpy as np


# modulation_types = ['8PSK', 'BPSK', 'QPSK', 'QAM16', 'QAM64', 'AM-DSB', 'AM-SSB', 'CPFSK', 'GFSK', 'WBFM', 'PAM4']
modulation_types = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
               '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM',
               '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
               'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']

def extract_snr_modulation_from_path(file_path):
    """
    从文件路径中提取SNR和调制方式，假设路径格式为 'test/modulation_type/SNR/data.npy'
    """
    parts = file_path.split(os.sep)
    modulation = parts[-3]  # 倒数第三个部分是调制方式
    snr = parts[-2]         # 倒数第二个部分是SNR文件夹
    return snr, modulation

class DualModalityDataset(Dataset):
    def __init__(self, npy_dir, is_train=True, transform=None):
        """
        npy_dir: 信号npy文件存放路径
        is_train: 如果为True，加载train数据，不区分SNR；否则加载test数据，区分SNR
        transform: 图像预处理操作
        """
        self.npy_dir = npy_dir
        self.is_train = is_train  # 是否为训练集
        self.transform = transform
        self.data = []

        # 处理train数据，并引入SNR数据（建议：可以按一定比例随机采样不同SNR）
        if is_train:
            for mod in modulation_types:
                npy_mod_path = os.path.join(npy_dir, mod)
                if os.path.isdir(npy_mod_path):
                    for npy_file in os.listdir(npy_mod_path):
                        if npy_file.endswith('.npy'):
                            npy_filepath = os.path.join(npy_mod_path, npy_file)
                            label = mod  # 用调制类型作为label
                            self.data.append((npy_filepath, label))

        # 处理test数据（区分SNR）
        else:
            for mod in modulation_types:
                npy_mod_path = os.path.join(npy_dir, mod)
                if os.path.isdir(npy_mod_path):
                    for snr_folder in os.listdir(npy_mod_path):
                        snr_path = os.path.join(npy_mod_path, snr_folder)
                        if os.path.isdir(snr_path):
                            for npy_file in os.listdir(snr_path):
                                if npy_file.endswith('.npy'):
                                    npy_filepath = os.path.join(snr_path, npy_file)
                                    label = mod  # 用调制类型作为label
                                    self.data.append((npy_filepath, label))

        # for mod in modulation_types:
        #     npy_mod_path = os.path.join(npy_dir, mod)
        #     for snr_folder in os.listdir(npy_mod_path):
        #         snr_path = os.path.join(npy_mod_path, snr_folder)
        #         for npy_file in os.listdir(snr_path):
        #             npy_filepath = os.path.join(snr_path, npy_file)
        #             label = mod  # 这里用调制类型作为label
        #             self.data.append((npy_filepath, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        npy_filepath, label = self.data[idx]
        signal = np.load(npy_filepath)  # 读取npy信号 (1024, 2)，包括I和Q信号
        signal = torch.tensor(signal.T, dtype=torch.float32)  # 转置为 (2, 1024)

        # 生成时频图
        spec = self.generate_spectrogram(signal)

        # 将标签转换为索引
        label_index = torch.tensor(self.get_label_index(label), dtype=torch.long)

        return signal, spec, label_index, npy_filepath

    def get_label_index(self, label):
        # 将调制方式转换为索引
        return modulation_types.index(label)


    def generate_spectrogram(self, signal):
        """通过包含 IQ 信号的数组动态生成时频图并返回numpy数组"""
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

        return torch.tensor(data, dtype=torch.float32).permute(2, 0, 1)  # 转换为 (3, 224, 224) 的 tensor


# 训练与测试日志文件路径
#
# class CustomDualModalityModel(nn.Module):
#     def __init__(self, num_classes):
#         super(CustomDualModalityModel, self).__init__()
#
#         # 序列分支 - 输入两路IQ信号，每路长度为1024
#         self.seq_branch = nn.Sequential(
#             nn.Linear(256, 128),  # 两路信号共 2048 个特征
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU()
#         )
#
#         # 图像分支 - 输入224x224x3的时频图像
#         self.img_conv = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),  # (224, 224, 16)
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),  # (112, 112, 16)
#
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # (112, 112, 32)
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),  # (56, 56, 32)
#
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # (56, 56, 64)
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),  # (28, 28, 64)
#
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # (28, 28, 128)
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),  # (14, 14, 128)
#         )
#
#         self.img_fc = nn.Sequential(
#             nn.Linear(128 * 14 * 14, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU()
#         )
#
#         # 融合层
#         self.fc_fusion = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, num_classes)  # 输出分类的种类数
#         )
#
#     def forward(self, iq_signal, img_input):
#         # 序列分支: 处理IQ信号
#         # 输入的iq_signal形状为 (batch_size, 2, 1024)，需要展平并组合两个通道
#         batch_size = iq_signal.size(0)
#         iq_signal = iq_signal.view(batch_size, -1)  # 变为 (batch_size, 2048) 256
#         seq_features = self.seq_branch(iq_signal)  # 输出大小为 (batch_size, 256)
#
#         # 图像分支: 处理时频图
#         # img_input = img_input.permute(0, 3, 1, 2)  # 将 (batch_size, 224, 224, 3) 转换为 (batch_size, 3, 224, 224)
#         img_features = self.img_conv(img_input)    # 卷积操作后维度为 (batch_size, 128, 14, 14)
#         img_features = img_features.view(batch_size, -1)  # 展平成 (batch_size, 128 * 14 * 14)
#         img_features = self.img_fc(img_features)   # 输出大小为 (batch_size, 256)
#
#         # 融合分支
#         fused_features = torch.cat((seq_features, img_features), dim=1)  # 拼接两个分支的特征 (batch_size, 512)
#         output = self.fc_fusion(fused_features)  # 分类输出 (batch_size, num_classes)
#
#         return output

class IQSignalModel(nn.Module):
    def __init__(self):
        super(IQSignalModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(256, 256)  # 修改为 256 以与融合层匹配

    def forward(self, x):
        x = self.cnn(x)
        x, _ = self.lstm(x.permute(0, 2, 1))  # LSTM expects (batch_size, seq_len, input_size)
        x = x[:, -1, :]  # Get the output from the last time step
        x = self.fc(x)
        return x

class CustomDualModalityModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomDualModalityModel, self).__init__()

        # 序列分支 - 使用IQSignalModel处理IQ信号
        self.seq_branch = IQSignalModel()

        # 图像分支 - 输入224x224x3的时频图像
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),  # (224, 224, 16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (112, 112, 16)

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # (112, 112, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (56, 56, 32)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # (56, 56, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (28, 28, 64)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # (28, 28, 128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (14, 14, 128)
        )

        self.img_fc = nn.Sequential(
            nn.Linear(128 * 14 * 14, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # 融合层
        self.fc_fusion = nn.Sequential(
            nn.Linear(256 + 256, 256),  # 注意输入特征的维度调整为256 + 256
            nn.ReLU(),
            nn.Linear(256, num_classes)  # 输出分类的种类数
        )

    def forward(self, iq_signal, img_input):
        # 序列分支: 处理IQ信号
        # 输入的iq_signal形状为 (batch_size, 2, 128)，需要调整为 (batch_size, 2, 1024) 以适应模型
        batch_size = iq_signal.size(0)
        iq_signal = iq_signal.view(batch_size, 2, 128)  # 假设输入长度为128
        seq_features = self.seq_branch(iq_signal)  # 输出大小为 (batch_size, 256)

        # 图像分支: 处理时频图
        # img_input = img_input.permute(0, 3, 1, 2)  # 将 (batch_size, 224, 224, 3) 转换为 (batch_size, 3, 224, 224)
        img_features = self.img_conv(img_input)    # 卷积操作后维度为 (batch_size, 128, 14, 14)
        img_features = img_features.view(batch_size, -1)  # 展平成 (batch_size, 128 * 14 * 14)
        img_features = self.img_fc(img_features)   # 输出大小为 (batch_size, 256)

        # 融合分支
        fused_features = torch.cat((seq_features, img_features), dim=1)  # 拼接两个分支的特征 (batch_size, 512)
        output = self.fc_fusion(fused_features)  # 分类输出 (batch_size, num_classes)

        return output


log_file = 'training_log.txt'

import time
import torch
from tqdm import tqdm

def train(model, device, train_loader, optimizer, criterion, epoch, log_interval=10):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} [Training]')  # 进度条
    for signals, images, labels, npy_path in progress_bar:
        signals = signals.to(device)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(signals, images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新进度条显示
        progress_bar.set_postfix({
            'loss': f'{running_loss / (len(train_loader)):.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    epoch_time = time.time() - start_time

    # 将结果保存到日志
    with open(log_file, 'a') as f:
        f.write(f'Epoch {epoch} [Training] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Time: {epoch_time:.2f}s\n')

    return epoch_acc  # 返回训练准确率

# def validate(model, device, test_loader, criterion, epoch):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     snr_correct = {}
#     snr_total = {}
#     start_time = time.time()
#
#     progress_bar = tqdm(test_loader, desc=f'Epoch {epoch} [Validation]')  # 进度条
#     with torch.no_grad():
#         for signals, images, labels, file_paths in progress_bar:  # 假设loader中包含文件路径信息
#             signals, images = signals.to(device), images.to(device)
#             labels = labels.to(device)
#
#             outputs = model(signals, images)
#             loss = criterion(outputs, labels)
#             test_loss += loss.item()
#
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#             # 从路径中提取SNR
#             for i, file_path in enumerate(file_paths):  # 假设file_paths是batch内文件的路径列表
#                 snr = extract_snr_from_path(file_path)
#                 if snr not in snr_correct:
#                     snr_correct[snr] = 0
#                     snr_total[snr] = 0
#                 snr_total[snr] += 1
#                 if predicted[i] == labels[i]:
#                     snr_correct[snr] += 1
#
#             # 更新进度条显示
#             progress_bar.set_postfix({
#                 'val_loss': f'{test_loss / len(test_loader):.4f}',
#                 'val_acc': f'{100 * correct / total:.2f}%'
#             })
#
#     val_loss = test_loss / len(test_loader)
#     val_acc = 100 * correct / total
#     val_time = time.time() - start_time
#
#     # 计算每种SNR下的准确率
#     snr_accs = {}
#     for snr in snr_correct:
#         snr_accs[snr] = 100 * snr_correct[snr] / snr_total[snr]
#
#     # 计算平均准确率
#     avg_snr_acc = sum(snr_accs.values()) / len(snr_accs) if snr_accs else 0
#
#     # 打印每种SNR的准确率
#     print(f'SNR-wise Accuracy:')
#     for snr, acc in snr_accs.items():
#         print(f'SNR {snr}: {acc:.2f}%')
#
#     print(f'Average SNR Accuracy: {avg_snr_acc:.2f}%')
#
#     # 将验证结果保存到日志
#     with open(log_file, 'a') as f:
#         f.write(f'Epoch {epoch} [Validation] - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%, Time: {val_time:.2f}s\n')
#         for snr, acc in snr_accs.items():
#             f.write(f'SNR {snr}: {acc:.2f}%\n')
#         f.write(f'Average SNR Accuracy: {avg_snr_acc:.2f}%\n')
#
#     return val_acc, avg_snr_acc  # 返回验证准确率和平均SNR准确率

def validate(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    mod_snr_correct = {}  # 用于存储不同调制方式下不同SNR的正确预测数
    mod_snr_total = {}    # 用于存储不同调制方式下不同SNR的总数
    start_time = time.time()

    # 定义 SNR 范围，按顺序排列
    snr_range = list(range(-20, 18, 2))

    # 确保每种调制方式和每种SNR的统计字典初始化
    for mod in modulation_types:
        mod_snr_correct[mod] = {snr: 0 for snr in snr_range}
        mod_snr_total[mod] = {snr: 0 for snr in snr_range}

    progress_bar = tqdm(test_loader, desc=f'Epoch {epoch} [Validation]')  # 进度条
    with torch.no_grad():
        for signals, images, labels, file_paths in progress_bar:  # 假设loader中包含文件路径信息
            signals, images = signals.to(device), images.to(device)
            labels = labels.to(device)

            outputs = model(signals, images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 从路径中提取 SNR 和调制方式
            for i, file_path in enumerate(file_paths):
                try:
                    snr, modulation = extract_snr_modulation_from_path(file_path)

                    # 确保 snr 是一个有效的整数值
                    snr_value = int(snr)

                    if modulation in mod_snr_total:
                        mod_snr_total[modulation][snr_value] += 1
                        if predicted[i] == labels[i]:
                            mod_snr_correct[modulation][snr_value] += 1
                except ValueError:
                    print(f"Warning: Invalid SNR value '{snr}' extracted from file path '{file_path}'")

            # 更新进度条显示
            progress_bar.set_postfix({
                'val_loss': f'{test_loss / len(test_loader):.4f}',
                'val_acc': f'{100 * correct / total:.2f}%'
            })

    val_loss = test_loss / len(test_loader)
    val_acc = 100 * correct / total
    val_time = time.time() - start_time

    # 计算每种调制方式和 SNR 的准确率
    mod_snr_accs = {}
    for modulation in mod_snr_correct:
        mod_snr_accs[modulation] = []
        for snr in snr_range:
            if mod_snr_total[modulation][snr] > 0:
                acc = 100 * mod_snr_correct[modulation][snr] / mod_snr_total[modulation][snr]
                mod_snr_accs[modulation].append(f'{acc:.2f}%')
            else:
                mod_snr_accs[modulation].append('--')  # 用占位符表示无数据

    # 计算平均准确率
    total_acc_sum = 0
    total_acc_count = 0
    for modulation in mod_snr_correct:
        for snr in snr_range:
            if mod_snr_total[modulation][snr] > 0:
                total_acc_sum += 100 * mod_snr_correct[modulation][snr] / mod_snr_total[modulation][snr]
                total_acc_count += 1

    avg_acc = total_acc_sum / total_acc_count if total_acc_count else 0

    # 打印每种调制方式和 SNR 的准确率
    print(f'Modulation-wise and SNR-wise Accuracy (SNRs from {snr_range[0]} to {snr_range[-1]}):')
    for modulation in mod_snr_accs:
        snr_acc_line = '\t'.join(mod_snr_accs[modulation])
        print(f'{modulation}:\t{snr_acc_line}')

    print(f'Average Accuracy: {avg_acc:.2f}%')

    # 将验证结果保存到日志
    with open(log_file, 'a') as f:
        f.write(f'Epoch {epoch} [Validation] - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%, Time: {val_time:.2f}s\n')
        f.write(f'Modulation-wise and SNR-wise Accuracy (SNRs from {snr_range[0]} to {snr_range[-1]}):\n')
        for modulation in mod_snr_accs:
            snr_acc_line = '\t'.join(mod_snr_accs[modulation])
            f.write(f'{modulation}:\t{snr_acc_line}\n')
        f.write(f'Average Accuracy: {avg_acc:.2f}%\n')

    return val_acc, avg_acc  # 返回验证准确率和平均准确率


def save_best_model(model, best_acc, current_acc, epoch):
    """保存表现最好的模型"""
    if current_acc > best_acc:
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Best model updated at epoch {epoch} with accuracy {current_acc:.2f}%")
        return current_acc
    return best_acc

def main():
    # 训练和测试数据集路径
    train_dir = './dataset2/train'
    test_dir = './dataset2/test'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda img: img[:3, :, :]),  # 只保留前3个通道（RGB）
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = DualModalityDataset(npy_dir=train_dir, is_train=True)
    test_dataset = DualModalityDataset(npy_dir=test_dir, is_train=False)

    # 创建数据加载器
    # train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomDualModalityModel(num_classes=24).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0  # 记录最优验证准确率

    val_acc = validate(model, device, test_loader, criterion, 5)

    # 清空日志文件
    with open(log_file, 'w') as f:
        f.write("Training Log\n")

    for epoch in range(1, 150):
        train_acc = train(model, device, train_loader, optimizer, criterion, epoch)

        # 每 10 轮验证一次
        if epoch % 1 == 0:
            val_acc, avg_snr_acc = validate(model, device, test_loader, criterion, epoch)
            best_acc = save_best_model(model, best_acc, avg_snr_acc, epoch)

if __name__ == '__main__':
    main()
