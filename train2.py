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


modulation_types = ['4ASK', 'BPSK', 'QPSK', '16QAM', '32QAM', 'AM-SSB-SC', 'AM-DSB-SC']

class DualModalityDataset(Dataset):
    def __init__(self, npy_dir, transform=None):
        """
        npy_dir: 信号npy文件存放路径, 其中每个子文件夹是调制类型，文件夹内是不同SNR文件夹
        transform: 图像预处理操作
        """
        self.npy_dir = npy_dir
        self.transform = transform
        self.data = []

        for mod in modulation_types:
            npy_mod_path = os.path.join(npy_dir, mod)
            for snr_folder in os.listdir(npy_mod_path):
                snr_path = os.path.join(npy_mod_path, snr_folder)
                for npy_file in os.listdir(snr_path):
                    npy_filepath = os.path.join(snr_path, npy_file)
                    label = mod  # 这里用调制类型作为label
                    self.data.append((npy_filepath, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        npy_filepath, label = self.data[idx]
        signal = np.load(npy_filepath)  # 读取npy信号 (1024, 2)，包括I和Q信号
        # signal = torch.tensor(signal.T, dtype=torch.float32)  # 转置为 (2, 1024)

        # 生成时频图
        spec = self.generate_spectrogram(signal)

        # 将标签转换为索引
        label_index = torch.tensor(self.get_label_index(label), dtype=torch.long)

        return signal, spec, label_index

    def get_label_index(self, label):
        # 将调制方式转换为索引
        return modulation_types.index(label)


    def generate_spectrogram(self, signal):
        """通过包含 IQ 信号的数组动态生成时频图并返回numpy数组"""
        # 提取 I 和 Q 信号
        I_signal = signal[:, 0]
        Q_signal = signal[:, 1]

        # 确保窗口大小小于信号长度
        window_size = 256
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

class CustomDualModalityModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomDualModalityModel, self).__init__()

        # 序列分支 - 输入两路IQ信号，每路长度为1024
        self.seq_branch = nn.Sequential(
            nn.Linear(2048, 1024),  # 两路信号共 2048 个特征
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

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
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)  # 输出分类的种类数
        )

    # def generate_spectrogram(self, signal):
    #     """通过包含 IQ 信号的数组动态生成时频图并返回numpy数组"""
    #     # 提取 I 和 Q 信号
    #     iq_signal = signal.cpu().numpy()
    #     I_signal = signal[:, 0]
    #     Q_signal = signal[:, 1]
    #
    #     # 计算时频图
    #     fs = 256  # 采样率，根据实际数据调整
    #     window_size = 128
    #     overlap = window_size - 1
    #     h = hamming(window_size)
    #     f, t, Sxx = spectrogram(I_signal + 1j * Q_signal, fs=fs, window=h, nperseg=window_size, noverlap=overlap)
    #     Sxx = np.fft.fftshift(Sxx, axes=0)  # 对频率轴进行FFT shift
    #
    #     # 对时频图进行归一化处理，确保在合理的数值范围
    #     Sxx = 10 * np.log10(Sxx + 1e-8)  # 转换为dB，避免log(0)的情况
    #
    #     # 归一化到0-255范围
    #     Sxx_min, Sxx_max = Sxx.min(), Sxx.max()
    #     Sxx_normalized = (Sxx - Sxx_min) / (Sxx_max - Sxx_min)  # 归一化到0-1
    #     Sxx_normalized = (Sxx_normalized * 255).astype(np.uint8)  # 再转为0-255范围的uint8
    #
    #     # 转换为3通道图像
    #     Sxx_3ch = np.stack([Sxx_normalized] * 3, axis=-1)
    #
    #     # 确保尺寸为224x224
    #     data = np.array(Image.fromarray(Sxx_3ch).resize((224, 224)))
    #
    #     return data

    def forward(self, iq_signal, img_input):
        # 序列分支: 处理IQ信号
        # 输入的iq_signal形状为 (batch_size, 2, 1024)，需要展平并组合两个通道
        batch_size = iq_signal.size(0)
        iq_signal = iq_signal.view(batch_size, -1)  # 变为 (batch_size, 2048)
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
    for signals, images, labels in progress_bar:
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


def validate(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    progress_bar = tqdm(test_loader, desc=f'Epoch {epoch} [Validation]')  # 进度条
    with torch.no_grad():
        for signals, images, labels in progress_bar:
            signals, images = signals.to(device), images.to(device)
            labels = labels.to(device)

            outputs = model(signals, images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新进度条显示
            progress_bar.set_postfix({
                'val_loss': f'{test_loss / len(test_loader):.4f}',
                'val_acc': f'{100 * correct / total:.2f}%'
            })

    val_loss = test_loss / len(test_loader)
    val_acc = 100 * correct / total
    val_time = time.time() - start_time

    # 将验证结果保存到日志
    with open(log_file, 'a') as f:
        f.write(f'Epoch {epoch} [Validation] - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%, Time: {val_time:.2f}s\n')

    return val_acc  # 返回验证准确率

def save_best_model(model, best_acc, current_acc, epoch):
    """保存表现最好的模型"""
    if current_acc > best_acc:
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Best model updated at epoch {epoch} with accuracy {current_acc:.2f}%")
        return current_acc
    return best_acc

def main():
    # 训练和测试数据集路径
    train_dir = 'train'
    test_dir = 'test'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda img: img[:3, :, :]),  # 只保留前3个通道（RGB）
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = DualModalityDataset(npy_dir=train_dir)
    test_dataset = DualModalityDataset(npy_dir=test_dir)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomDualModalityModel(num_classes=24).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0  # 记录最优验证准确率

    # 清空日志文件
    with open(log_file, 'w') as f:
        f.write("Training Log\n")

    for epoch in range(1, 21):
        train_acc = train(model, device, train_loader, optimizer, criterion, epoch)

        # 每 10 轮验证一次
        if epoch % 1 == 0:
            val_acc = validate(model, device, test_loader, criterion, epoch)
            best_acc = save_best_model(model, best_acc, val_acc, epoch)

if __name__ == '__main__':
    main()
