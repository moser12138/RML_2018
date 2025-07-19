# 在使用代码将RML_2018数据集导出为.npy文件后，到处信号存储在IQ_signals_npy文件夹目录下，目录中按照信号调制类型分为不同文件夹，在不同调制类型文件夹中按照snr数值分为不同文件夹，请帮我写一个完整的python信号分类代码，要求不使用官方模型，自己编写一个完整的双模态分类模型，一种输入类型为信号的npy文件，另一个输入为通过信号npy文件画出的信号时频图，时频图尺寸规定为224*224，两路输入后，进入蛇毒学习网络训练，并最后特征融合，共同决定分类标签，请打包函数，写出main函数，分别写出读取数据集，训练，测试，等功能，完整的进行训练
# 请改进其中的train与test函数，要求训练过程中每10轮会验证一次当前模型参数的分类准确率，并与已经保存的参数准确率比较，如果当前准确率较高，则会替换保存的参数，并且训练和测试过程中会用tqdm中进度条方式展示训练进度，每结束一轮，会在log文件中保存当前轮训练结果，10轮验证时也会保存结果，训练中需要展示loss，准确率，一轮花费时间等参数
# 请基于以下代码修改，将绘制时频图代码写在神经网络中，输入网络的为两路IQ信号，先将IQ信号转化为时频图，输入图像模态网络分支，再将IQ信号序列输入序列网络分支进行综合分类操纵，IQ信号尺度分为两路，每路长度1024,请将其转为224*224的时频图

import torch
import torch.nn as nn
import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, hamming
import time
import numpy as np
import io
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm  # 用于显示进度条
from torch.utils.data import Dataset

# modulation_types = ['8PSK', 'BPSK', 'QPSK', 'QAM16', 'QAM64', 'AM-DSB', 'AM-SSB', 'CPFSK', 'GFSK', 'WBFM', 'PAM4']
modulation_types = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
               '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM',
               '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
               'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
log_file = 'training_log.txt'

def extract_snr_modulation_from_path(file_path):
    """
    从文件路径中提取SNR和调制方式，假设路径格式为 'test/modulation_type/SNR/data.npy'
    """
    parts = file_path.split(os.sep)
    modulation = parts[-3]  # 倒数第三个部分是调制方式
    snr = parts[-2]         # 倒数第二个部分是SNR文件夹
    return snr, modulation

class DualModalityDataset(Dataset):
    def __init__(self, npy_dir, is_train='train', transform=None):
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
        if is_train == 'train':
            signal_dir = os.path.join(npy_dir, 'signal')
            gaf_dir = os.path.join(npy_dir, 'geram')
            for mod in modulation_types:
                signal_mod_path = os.path.join(signal_dir, mod)
                gaf_mod_path = os.path.join(gaf_dir, mod)
                if os.path.isdir(signal_mod_path):
                    for npy_file in os.listdir(signal_mod_path):
                        if npy_file.endswith('.npy'):
                            signal_filepath = os.path.join(signal_mod_path, npy_file)
                            gaf_filepath = os.path.join(gaf_mod_path, npy_file)
                            label = mod  # 用调制类型作为label
                            self.data.append((signal_filepath, gaf_filepath, label))

            # 处理test数据（区分SNR）
        elif is_train == 'test':
            signal_dir = os.path.join(npy_dir, 'signal')
            gaf_dir = os.path.join(npy_dir, 'geram')
            for mod in modulation_types:
                signal_mod_path = os.path.join(signal_dir, mod)
                gaf_mod_path = os.path.join(gaf_dir, mod)
                if os.path.isdir(signal_mod_path):
                    for snr_folder in os.listdir(signal_mod_path):
                        signal_snr_path = os.path.join(signal_mod_path, snr_folder)
                        gaf_snr_path = os.path.join(gaf_mod_path, snr_folder)
                        if os.path.isdir(signal_snr_path):
                            for npy_file in os.listdir(signal_snr_path):
                                if npy_file.endswith('.npy'):
                                    signal_filepath = os.path.join(signal_snr_path, npy_file)
                                    gaf_filepath = os.path.join(gaf_snr_path, npy_file)
                                    label = mod  # 用调制类型作为label
                                    self.data.append((signal_filepath, gaf_filepath, label))
        else:
            signal_dir = os.path.join(npy_dir, 'signal2')
            gaf_dir = os.path.join(npy_dir, 'geram2')
            for mod in modulation_types:
                signal_mod_path = os.path.join(signal_dir, mod)
                gaf_mod_path = os.path.join(gaf_dir, mod)
                if os.path.isdir(signal_mod_path):
                    for snr_folder in os.listdir(signal_mod_path):
                        signal_snr_path = os.path.join(signal_mod_path, snr_folder)
                        gaf_snr_path = os.path.join(gaf_mod_path, snr_folder)
                        if os.path.isdir(signal_snr_path):
                            for npy_file in os.listdir(signal_snr_path):
                                if npy_file.endswith('.npy'):
                                    signal_filepath = os.path.join(signal_snr_path, npy_file)
                                    gaf_filepath = os.path.join(gaf_snr_path, npy_file)
                                    label = mod  # 用调制类型作为label
                                    self.data.append((signal_filepath, gaf_filepath, label))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        npy_filepath, gaf_filepath, label = self.data[idx]
        # 获取IQ
        signal = np.load(npy_filepath)  # 读取npy信号 (1024, 2)，包括I和Q信号

        # 获取gaf
        gaf = np.load(gaf_filepath)
        # 将GAF图像转换为浮点类型，并归一化到[0, 1]
        gaf = gaf.astype(np.float32)

        # 将标签转换为索引
        label_index = torch.tensor(self.get_label_index(label), dtype=torch.long)

        return signal, gaf, label_index, npy_filepath

    def get_label_index(self, label):
        # 将调制方式转换为索引
        return modulation_types.index(label)

# 原始模型

# class IQSignalModel(nn.Module):
#     def __init__(self):
#         super(IQSignalModel, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv1d(in_channels=2, out_channels=64, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2),
#             nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2)
#         )
#         self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True, dropout=0.5)
#         self.fc = nn.Linear(256, 256)  # 修改为 256 以与融合层匹配
#
#     def forward(self, x):
#         x = self.cnn(x)
#         x, _ = self.lstm(x.permute(0, 2, 1))  # LSTM expects (batch_size, seq_len, input_size)
#         x = x[:, -1, :]  # Get the output from the last time step
#         x = self.fc(x)
#         return x
#
# class CustomDualModalityModel(nn.Module):
#     def __init__(self, num_classes):
#         super(CustomDualModalityModel, self).__init__()
#
#         # 序列分支 - 使用IQSignalModel处理IQ信号
#         self.seq_branch = IQSignalModel()
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
#             nn.Linear(256 + 256, 256),  # 注意输入特征的维度调整为256 + 256
#             nn.ReLU(),
#             nn.Linear(256, num_classes)  # 输出分类的种类数
#         )
#
#     def forward(self, iq_signal, img_input):
#         # 序列分支: 处理IQ信号
#         # 输入的iq_signal形状为 (batch_size, 2, 128)，需要调整为 (batch_size, 2, 1024) 以适应模型
#         batch_size = iq_signal.size(0)
#         iq_signal = iq_signal.view(batch_size, 2, 1024)  # 假设输入长度为128
#         seq_features = self.seq_branch(iq_signal)  # 输出大小为 (batch_size, 256)
#
#         # 图像分支: 处理时频图
#         img_input = img_input.permute(0, 3, 1, 2)  # 将 (batch_size, 224, 224, 3) 转换为 (batch_size, 3, 224, 224)
#         img_features = self.img_conv(img_input)    # 卷积操作后维度为 (batch_size, 128, 14, 14)
#         img_features = img_features.reshape(batch_size, -1)  # 展平成 (batch_size, 128 * 14 * 14)
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
            nn.Dropout(0.15),  # 添加 Dropout 层

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.15)   # 添加 Dropout 层
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True, dropout=0.5)
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2)  # 添加 Dropout 层
        )

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
            nn.Dropout(0.15),  # 添加 Dropout 层

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # (112, 112, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (56, 56, 32)
            nn.Dropout(0.15),  # 添加 Dropout 层

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # (56, 56, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (28, 28, 64)
            nn.Dropout(0.15),  # 添加 Dropout 层

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # (28, 28, 128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (14, 14, 128)
            nn.Dropout(0.15)  # 添加 Dropout 层
        )

        self.img_fc = nn.Sequential(
            nn.Linear(128 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加 Dropout 层

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)  # 添加 Dropout 层
        )

        # 添加可学习的权重参数
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始化信号分支权重
        self.beta = nn.Parameter(torch.tensor(0.5))   # 初始化图像分支权重

        # 融合层
        self.fc_fusion = nn.Sequential(
            nn.Linear(256, 256),  # 融合后的特征维度为256
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加 Dropout 层
            nn.Linear(256, num_classes)  # 输出分类的种类数
        )

    def forward(self, iq_signal, img_input):
        # 序列分支: 处理IQ信号
        batch_size = iq_signal.size(0)
        iq_signal = iq_signal.view(batch_size, 2, 1024)  # 假设输入长度为1024
        seq_features = self.seq_branch(iq_signal)  # 输出大小为 (batch_size, 256)

        # 图像分支: 处理时频图
        img_input = img_input.permute(0, 3, 1, 2)  # 将 (batch_size, 224, 224, 3) 转换为 (batch_size, 3, 224, 224)
        img_features = self.img_conv(img_input)    # 卷积操作后维度为 (batch_size, 128, 14, 14)
        img_features = img_features.reshape(batch_size, -1)  # 展平成 (batch_size, 128 * 14 * 14)
        img_features = self.img_fc(img_features)   # 输出大小为 (batch_size, 256)

        # 融合分支 - 通过alpha和beta进行自学习权重融合
        seq_weighted = self.alpha * seq_features  # 乘以可学习的alpha权重
        img_weighted = self.beta * img_features   # 乘以可学习的beta权重
        fused_features = seq_weighted + img_weighted  # 加和两者 (batch_size, 256)

        # 进行分类
        output = self.fc_fusion(fused_features)  # 分类输出 (batch_size, num_classes)

        return output



def train(model, device, train_loader, optimizer, criterion, epoch, log_interval=10):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} [Training]')  # 进度条
    for signals, images, labels, npy_path in progress_bar:
        signals = signals.to(device).float()
        images = images.to(device).float()
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

def validate(model, device, test_loader, criterion, epoch, model_path = None):
    # 如果提供了模型路径，则加载模型参数
    if model_path:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model parameters from {model_path}")

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    mod_snr_correct = {}  # 用于存储不同调制方式下不同SNR的正确预测数
    mod_snr_total = {}    # 用于存储不同调制方式下不同SNR的总数
    start_time = time.time()

    # 定义 SNR 范围，按顺序排列
    snr_range = list(range(-20, 12, 2))

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

    # 计算每个 SNR 下的所有信号种类的准确率
    snr_accuracy = {}
    for snr in snr_range:
        snr_correct = 0
        snr_total = 0
        for mod in modulation_types:
            snr_correct += mod_snr_correct[mod][snr]
            snr_total += mod_snr_total[mod][snr]
        if snr_total > 0:
            snr_accuracy[snr] = 100 * snr_correct / snr_total
        else:
            snr_accuracy[snr] = None

    # 打印每种调制方式和 SNR 的准确率
    print(f'Modulation-wise and SNR-wise Accuracy (SNRs from {snr_range[0]} to {snr_range[-1]}):')
    for modulation in mod_snr_accs:
        snr_acc_line = '\t'.join(mod_snr_accs[modulation])
        print(f'{modulation}:\t{snr_acc_line}')

    print(f'Average Accuracy: {avg_acc:.2f}%')

    # 打印每个 SNR 下所有信号种类的准确率
    print(f'Overall Accuracy for each SNR:')
    overall_snr_acc_line = []
    for snr in snr_range:
        if snr_accuracy[snr] is not None:
            overall_snr_acc_line.append(f'{snr}:{snr_accuracy[snr]:.2f}%')
        else:
            overall_snr_acc_line.append(f'SNR {snr}: --')
    overall_snr_acc_line_str = '\t'.join(overall_snr_acc_line)
    print(overall_snr_acc_line_str)

    # 将验证结果保存到日志
    with open(log_file, 'a') as f:
        f.write(f'Epoch {epoch} [Validation] - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%, Time: {val_time:.2f}s\n')
        f.write(f'Modulation-wise and SNR-wise Accuracy (SNRs from {snr_range[0]} to {snr_range[-1]}):\n')
        for modulation in mod_snr_accs:
            snr_acc_line = '\t'.join(mod_snr_accs[modulation])
            f.write(f'{modulation}:\t{snr_acc_line}\n')
        f.write(f'Average Accuracy: {avg_acc:.2f}%\n')

        # 保存每个 SNR 下所有信号种类的准确率
        f.write(f'\nOverall Accuracy for each SNR:\n')
        f.write(overall_snr_acc_line_str + '\n')

    return val_acc, avg_acc  # 返回验证准确率和平均准确率

def test_train_loader_accuracy(model, device, train_loader, model_path='best_model_grame.pth'):
    # 如果提供了模型路径，则加载模型参数
    if model_path:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model parameters from {model_path}")


    model.eval()
    correct = 0
    total = 0
    mod_correct = {}  # 用于存储每种调制方式的正确预测数
    mod_total = {}    # 用于存储每种调制方式的总数

    # 初始化每种调制方式的统计字典
    for mod in modulation_types:
        mod_correct[mod] = 0
        mod_total[mod] = 0

    with torch.no_grad():
        for signals, images, labels, _ in train_loader:
            signals, images = signals.to(device), images.to(device)
            labels = labels.to(device)

            # 预测输出
            outputs = model(signals, images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 统计每种调制方式的正确预测数
            for i, label in enumerate(labels):
                label_str = modulation_types[label.item()]  # 将数字标签转换为调制类型字符串
                mod_total[label_str] += 1
                if predicted[i] == label:
                    mod_correct[label_str] += 1

    # 计算每种调制方式的准确率
    mod_accs = {}
    for mod in modulation_types:
        if mod_total[mod] > 0:
            mod_accs[mod] = 100 * mod_correct[mod] / mod_total[mod]
        else:
            mod_accs[mod] = 0

    # 计算平均准确率
    avg_acc = 100 * correct / total if total > 0 else 0

    # 打印每种调制方式的准确率
    print('Modulation-wise Accuracy:')
    for mod, acc in mod_accs.items():
        print(f'{mod}: {acc:.2f}%')

    print(f'Average Accuracy: {avg_acc:.2f}%')

    return mod_accs, avg_acc  # 返回每种调制方式的准确率和平均准确率

def save_best_model(model, best_acc, current_acc, epoch):
    """保存表现最好的模型"""
    if current_acc > best_acc:
        torch.save(model.state_dict(), 'best_model_grame.pth')
        print(f"Best model updated at epoch {epoch} with accuracy {current_acc:.2f}%")
        return current_acc
    return best_acc

def main():
    # 训练和测试数据集路径
    train_dir = 'dataset/train'
    test_dir = 'dataset/test'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda img: img[:3, :, :]),  # 只保留前3个通道（RGB）
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = DualModalityDataset(npy_dir=train_dir, is_train='Train')
    test_dataset = DualModalityDataset(npy_dir=test_dir, is_train='test')
    val_dataset = DualModalityDataset(npy_dir=train_dir, is_train='val')

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomDualModalityModel(num_classes=24).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0  # 记录最优验证准确率

    # validate(model, device, val_loader, criterion, 5, model_path='best_model_grame.pth')
    # validate(model, device, test_loader, criterion, 5, model_path='best_model_grame.pth')
    # test_train_loader_accuracy(model, device, train_loader, model_path='best_model_grame.pth')

    # 清空日志文件
    with open(log_file, 'w') as f:
        f.write("Training Log\n")

    for epoch in range(1, 101):
        train_acc = train(model, device, train_loader, optimizer, criterion, epoch)

        # 每 10 轮验证一次
        if epoch % 5 == 0:
            val_acc, avg_snr_acc = validate(model, device, test_loader, criterion, epoch)
            best_acc = save_best_model(model, best_acc, avg_snr_acc, epoch)

if __name__ == '__main__':
    main()
